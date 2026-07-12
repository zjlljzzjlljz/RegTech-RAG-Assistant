#!/usr/bin/env python
"""Retrieval evaluation script for compliance RAG pipeline.

Usage:
    python -m src.evaluation.eval_retrieval
    python -m src.evaluation.eval_retrieval --fusion rrf
    python -m src.evaluation.eval_retrieval --fusion dedup
    python -m src.evaluation.eval_retrieval --fusion rrf --with-rerank
    python -m src.evaluation.eval_retrieval --fusion sparse-only
    python -m src.evaluation.eval_retrieval --fusion dense-only

Performs direct hybrid search (bypassing async/LLM pipeline),
and reports Hit@1, Hit@3, Hit@5, and MRR.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from config.settings import configure_logging, get_settings
from src.evaluation.metrics import calculate_hit_at_k, calculate_mrr
from src.indexing import BGEM3EmbeddingClient, MilvusHybridStore

logger = logging.getLogger(__name__)

_OUTPUT_FIELDS = [
    "chunk_id",
    "parent_id",
    "chunk_type",
    "source_file",
    "page_number",
    "text",
    "metadata_json",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only evaluation over labeled queries.")
    parser.add_argument(
        "--queries-file",
        default=str(Path(__file__).resolve().parent / "test_queries_v2.json"),
        help="Path to a JSON file with query, text, and ground_truth_chunk_ids entries.",
    )
    parser.add_argument(
        "--suite",
        choices=["single", "all"],
        default="single",
        help="Evaluate one query file or the combined EN/ZH/HK-mixed 75-query suite.",
    )
    parser.add_argument("--output", type=Path, help="Write machine-readable metrics JSON for CI gating.")
    parser.add_argument(
        "--fusion",
        default="rrf",
        choices=["all", "weighted-sweep", "rrf", "dedup", "sparse-only", "dense-only"],
        help=(
            "Fusion strategy for combining dense and sparse retrieval arms:\n"
            "  all        — run all four ablations with one embedding per query\n"
            "  weighted-sweep — scan dense:sparse RRF weights with one embedding per query\n"
            "  rrf        — Reciprocal Rank Fusion (score-based, recommended)\n"
            "  dedup      — dense-first then sparse dedup append (original broken behaviour)\n"
            "  sparse-only — pure sparse retrieval only\n"
            "  dense-only  — pure dense retrieval only"
        ),
    )
    parser.add_argument(
        "--with-rerank",
        action="store_true",
        help="Apply BAAI/bge-reranker-large cross-encoder reranking after fusion.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Explicitly disable reranking (RRF-only diagnostic mode).",
    )
    return parser.parse_args(argv)


def load_test_queries(filepath: Path) -> list[dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(filepath, encoding="utf-8") as fh:
        queries = json.load(fh)
    logger.info("Loaded %d test queries from %s", len(queries), filepath)
    return queries


def _reciprocal_rank_fusion(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    rrf_k: int,
    rrf_top_k: int,
) -> list[dict[str, Any]]:
    """RRF fusion — identical to ComplianceRetrievalPipeline._reciprocal_rank_fusion."""
    return _weighted_rank_fusion(dense_hits, sparse_hits, rrf_k, rrf_top_k, 1.0, 1.0)


def _weighted_rank_fusion(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    rrf_k: int,
    rrf_top_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[dict[str, Any]]:
    """Weighted RRF over dense and sparse rankings."""
    scored: dict[str, float] = {}
    canonical: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        cid = str(hit["chunk_id"])
        scored[cid] = scored.get(cid, 0.0) + dense_weight / (rrf_k + rank)
        canonical.setdefault(cid, hit)
    for rank, hit in enumerate(sparse_hits, start=1):
        cid = str(hit["chunk_id"])
        scored[cid] = scored.get(cid, 0.0) + sparse_weight / (rrf_k + rank)
        canonical.setdefault(cid, hit)
    merged = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return [canonical[cid] for cid, _ in merged[:rrf_top_k]]


def _dense_first_dedup(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for hit in [*dense_hits, *sparse_hits]:
        chunk_id = str(hit.get("chunk_id", ""))
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            merged.append(hit)
    return merged


def search_single_query(
    query: str,
    embedding_client: BGEM3EmbeddingClient,
    store: MilvusHybridStore,
    settings: Any,
    fusion: str = "rrf",
    with_rerank: bool = False,
    reranker: Any | None = None,
) -> list[str]:
    """Direct hybrid search — returns chunk_id list.

    fusion modes:
        rrf         — RRF fusion of dense[:dense_top_k] and sparse[:sparse_top_k]
        dedup       — dense[:top_k] then sparse[:top_k] with dedup (original broken behaviour)
        sparse-only — pure sparse retrieval only
        dense-only  — pure dense retrieval only

    when with_rerank=True, RRF/dedup/dense-only results are reranked via
    BAAI/bge-reranker-large cross-encoder before returning.
    """
    embedding = embedding_client.encode(
        query,
        prompt="Represent this sentence for searching relevant passages: ",
    )
    rrf_k = settings.retrieval.rrf_k
    rrf_top_k = settings.retrieval.rrf_top_k
    rerank_top_k = settings.retrieval.rerank_top_k
    rerank_threshold = settings.retrieval.rerank_score_threshold
    dense_top_k = settings.retrieval.dense_top_k
    sparse_top_k = settings.retrieval.sparse_top_k

    # ------------------------------------------------------------------
    # Retrieval arm selection
    # ------------------------------------------------------------------
    if fusion == "sparse-only":
        sparse_hits = store.sparse_only_search(
            sparse_vector=embedding.sparse_vector,
            top_k=rerank_top_k if with_rerank else rrf_top_k,
            output_fields=_OUTPUT_FIELDS,
            filters='chunk_type == "child"',
        )
        hits: list[dict[str, Any]] = sparse_hits

    elif fusion == "dense-only":
        dense_hits = store.dense_only_search(
            dense_vector=embedding.dense_vector,
            top_k=rerank_top_k if with_rerank else rrf_top_k,
            output_fields=_OUTPUT_FIELDS,
            filters='chunk_type == "child"',
        )
        hits = dense_hits

    else:
        # hybrid: dense + sparse then fuse
        dense_hits, sparse_hits = store.hybrid_search(
            dense_vector=embedding.dense_vector,
            sparse_vector=embedding.sparse_vector,
            top_k=max(dense_top_k, sparse_top_k),
            output_fields=_OUTPUT_FIELDS,
            filters='chunk_type == "child"',
        )
        dense_trunc = dense_hits[:dense_top_k]
        sparse_trunc = sparse_hits[:sparse_top_k]

        if fusion == "dedup":
            hits = _dense_first_dedup(dense_trunc, sparse_trunc)

        else:  # fusion == "rrf"
            hits = _reciprocal_rank_fusion(dense_trunc, sparse_trunc, rrf_k, rrf_top_k)

    # ------------------------------------------------------------------
    # Reranking (applies to all fusion modes except sparse-only when reranking)
    # ------------------------------------------------------------------
    if with_rerank:
        # Build RetrievedChunk list from hits
        from dataclasses import dataclass

        @dataclass
        class RetrievedChunk:
            chunk_id: str
            parent_id: Any
            text: str
            source_file: str
            page_number: Any
            chunk_type: str
            score: float
            metadata: dict

        chunks = [
            RetrievedChunk(
                chunk_id=str(hit["chunk_id"]),
                parent_id=hit.get("parent_id"),
                text=str(hit.get("text", "")),
                source_file=str(hit.get("source_file", "")),
                page_number=hit.get("page_number"),
                chunk_type=str(hit.get("chunk_type", "")),
                score=float(hit.get("score", 0.0)),
                metadata=dict(hit.get("metadata", {})),
            )
            for hit in hits
        ]
        active_reranker = reranker
        if active_reranker is None:
            from src.retrieval.query_pipeline import CrossEncoderRerankerClient

            active_reranker = CrossEncoderRerankerClient(settings.inference.reranker_model_name)
        reranked = active_reranker.rerank(query, chunks, rerank_top_k, rerank_threshold)
        return [c.chunk_id for c in reranked]

    # ------------------------------------------------------------------
    # No rerank — return raw hits as chunk_id list
    # ------------------------------------------------------------------
    chunk_ids: list[str] = []
    for hit in hits:
        cid = str(hit.get("chunk_id", ""))
        if cid:
            chunk_ids.append(cid)
    return chunk_ids


def evaluate(
    test_queries: list[dict[str, Any]],
    embedding_client: BGEM3EmbeddingClient,
    store: MilvusHybridStore,
    settings: Any,
    fusion: str = "rrf",
    with_rerank: bool = False,
) -> list[dict[str, Any]]:
    """Run all test queries through direct hybrid search."""
    reranker = None
    if with_rerank:
        from src.retrieval.query_pipeline import CrossEncoderRerankerClient

        reranker = CrossEncoderRerankerClient(settings.inference.reranker_model_name)

    results: list[dict[str, Any]] = []
    for i, tq in enumerate(test_queries):
        query = tq.get("query", "")
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(test_queries), query[:80])
        try:
            retrieved_ids = search_single_query(
                query,
                embedding_client,
                store,
                settings,
                fusion=fusion,
                with_rerank=with_rerank,
                reranker=reranker,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Search failed for query '%s': %s", query[:60], exc)
            retrieved_ids = []

        results.append(
            {
                "query": query,
                "text": tq.get("text", query),
                "ground_truth_chunk_ids": tq.get("ground_truth_chunk_ids", []),
                "retrieved_chunk_ids": retrieved_ids,
            }
        )
    return results


def evaluate_ablation(
    test_queries: list[dict[str, Any]],
    embedding_client: BGEM3EmbeddingClient,
    store: MilvusHybridStore,
    settings: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate four retrieval modes while encoding each query exactly once."""
    mode_results: dict[str, list[dict[str, Any]]] = {
        "dense_only": [],
        "sparse_only": [],
        "dedup": [],
        "rrf": [],
    }
    dense_top_k = settings.retrieval.dense_top_k
    sparse_top_k = settings.retrieval.sparse_top_k
    rrf_top_k = settings.retrieval.rrf_top_k

    for index, test_query in enumerate(test_queries, start=1):
        query = str(test_query.get("query", ""))
        logger.info("[%d/%d] Ablation retrieval: %s", index, len(test_queries), query[:80])
        try:
            embedding = embedding_client.encode(
                query,
                prompt="Represent this sentence for searching relevant passages: ",
            )
            dense_hits, sparse_hits = store.hybrid_search(
                dense_vector=embedding.dense_vector,
                sparse_vector=embedding.sparse_vector,
                top_k=max(dense_top_k, sparse_top_k),
                output_fields=_OUTPUT_FIELDS,
                filters='chunk_type == "child"',
            )
            dense_trunc = dense_hits[:dense_top_k]
            sparse_trunc = sparse_hits[:sparse_top_k]
            rankings = {
                "dense_only": dense_hits[:rrf_top_k],
                "sparse_only": sparse_hits[:rrf_top_k],
                "dedup": _dense_first_dedup(dense_trunc, sparse_trunc),
                "rrf": _reciprocal_rank_fusion(
                    dense_trunc,
                    sparse_trunc,
                    settings.retrieval.rrf_k,
                    rrf_top_k,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Ablation search failed for '%s': %s", query[:60], exc)
            rankings = {mode: [] for mode in mode_results}

        for mode, hits in rankings.items():
            mode_results[mode].append(
                {
                    "query": query,
                    "ground_truth_chunk_ids": test_query.get("ground_truth_chunk_ids", []),
                    "retrieved_chunk_ids": [str(hit.get("chunk_id", "")) for hit in hits],
                }
            )
    return mode_results


def evaluate_weight_sweep(
    test_queries: list[dict[str, Any]],
    embedding_client: BGEM3EmbeddingClient,
    store: MilvusHybridStore,
    settings: Any,
    weights: tuple[tuple[float, float], ...] = (
        (10.0, 1.0),
        (8.0, 1.0),
        (6.0, 1.0),
        (5.0, 1.0),
        (4.0, 1.0),
        (3.0, 1.0),
        (2.0, 1.0),
        (1.0, 1.0),
    ),
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate weighted RRF variants while encoding every query once."""
    mode_results = {
        f"dense_{dense_weight:g}_sparse_{sparse_weight:g}": []
        for dense_weight, sparse_weight in weights
    }
    dense_top_k = settings.retrieval.dense_top_k
    sparse_top_k = settings.retrieval.sparse_top_k

    for index, test_query in enumerate(test_queries, start=1):
        query = str(test_query.get("query", ""))
        logger.info("[%d/%d] Weighted RRF retrieval: %s", index, len(test_queries), query[:80])
        try:
            embedding = embedding_client.encode(
                query,
                prompt="Represent this sentence for searching relevant passages: ",
            )
            dense_hits, sparse_hits = store.hybrid_search(
                dense_vector=embedding.dense_vector,
                sparse_vector=embedding.sparse_vector,
                top_k=max(dense_top_k, sparse_top_k),
                output_fields=_OUTPUT_FIELDS,
                filters='chunk_type == "child"',
            )
            dense_trunc = dense_hits[:dense_top_k]
            sparse_trunc = sparse_hits[:sparse_top_k]
            rankings = {
                f"dense_{dense_weight:g}_sparse_{sparse_weight:g}": _weighted_rank_fusion(
                    dense_trunc,
                    sparse_trunc,
                    settings.retrieval.rrf_k,
                    settings.retrieval.rrf_top_k,
                    dense_weight,
                    sparse_weight,
                )
                for dense_weight, sparse_weight in weights
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Weighted RRF search failed for '%s': %s", query[:60], exc)
            rankings = {mode: [] for mode in mode_results}

        for mode, hits in rankings.items():
            mode_results[mode].append(
                {
                    "query": query,
                    "ground_truth_chunk_ids": test_query.get("ground_truth_chunk_ids", []),
                    "retrieved_chunk_ids": [str(hit.get("chunk_id", "")) for hit in hits],
                }
            )
    return mode_results


def build_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "hit_at_1": calculate_hit_at_k(results, 1),
        "hit_at_3": calculate_hit_at_k(results, 3),
        "hit_at_5": calculate_hit_at_k(results, 5),
        "recall_at_10": calculate_hit_at_k(results, 10),
        "mrr": calculate_mrr(results),
    }


def print_report(results: list[dict[str, Any]], fusion: str, with_rerank: bool) -> None:
    """Print evaluation report to stdout."""
    mode_tag = f"[{fusion.upper()}" + (", RERANKED]" if with_rerank else "]")
    print("\n" + "=" * 60)
    print(f"  Compliance Retrieval Evaluation Report {mode_tag}")
    print("=" * 60)
    print(f"  Total queries evaluated: {len(results)}")

    metrics = build_metrics(results)
    hit1 = metrics["hit_at_1"]
    hit3 = metrics["hit_at_3"]
    hit5 = metrics["hit_at_5"]
    mrr = metrics["mrr"]

    print(f"\n  Hit@1:  {hit1:.4f}")
    print(f"  Hit@3:  {hit3:.4f}")
    print(f"  Hit@5:  {hit5:.4f}")
    print(f"  MRR:    {mrr:.4f}")

    annotated_count = sum(1 for r in results if r["ground_truth_chunk_ids"])
    if annotated_count > 0:
        print(f"\n  Queries with ground truth: {annotated_count}/{len(results)}")
    else:
        print("\n  (No ground truth annotations — run annotate_gt.py first)")

    print("\n" + "-" * 60)
    print("  Per-Query Detail")
    print("-" * 60)
    for i, r in enumerate(results):
        gt = set(r["ground_truth_chunk_ids"])
        retrieved = r["retrieved_chunk_ids"][:5]
        hits = gt & set(retrieved) if gt else set()
        status = "✓" if hits else ("·" if not gt else "✗")
        print(f"  [{status}] Q{i + 1}: {r['query'][:70]}")
        if gt:
            print(f"         GT: {list(gt)[:5]}")
            print(f"         Retrieved (top-5): {retrieved}")
        else:
            print(f"         (no ground truth — skipped in scoring)")

    print("\n" + "=" * 60)
    logger.info(
        "Eval complete [%s] — Hit@1=%.4f Hit@3=%.4f Hit@5=%.4f MRR=%.4f",
        mode_tag,
        hit1,
        hit3,
        hit5,
        mrr,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.app.log_level)

    queries_path = Path(args.queries_file)
    if not queries_path.exists():
        logger.error("Test queries file not found: %s", queries_path)
        raise SystemExit(1)

    if args.suite == "all":
        suite_dir = Path(__file__).resolve().parent
        suite_paths = [
            suite_dir / "test_queries_v2.json",
            suite_dir / "test_queries_en_v2.json",
            suite_dir / "test_queries_hk_mixed_v2.json",
        ]
        test_queries = []
        for path in suite_paths:
            test_queries.extend(load_test_queries(path))
    else:
        test_queries = load_test_queries(queries_path)

    store = MilvusHybridStore(settings)
    embedding_client = BGEM3EmbeddingClient(settings.inference.embedding_model_name)

    # --no-rerank overrides --with-rerank
    with_rerank = bool(args.with_rerank) and not bool(args.no_rerank)

    if args.fusion in {"all", "weighted-sweep"}:
        if with_rerank:
            raise SystemExit(f"--fusion {args.fusion} cannot be combined with --with-rerank")
        mode_results = (
            evaluate_ablation(test_queries, embedding_client, store, settings)
            if args.fusion == "all"
            else evaluate_weight_sweep(test_queries, embedding_client, store, settings)
        )
        metrics = {mode: build_metrics(results) for mode, results in mode_results.items()}
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        results = evaluate(
            test_queries,
            embedding_client,
            store,
            settings,
            fusion=args.fusion,
            with_rerank=with_rerank,
        )
        print_report(results, fusion=args.fusion, with_rerank=with_rerank)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(build_metrics(results), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
