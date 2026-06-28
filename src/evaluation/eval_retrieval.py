#!/usr/bin/env python
"""Retrieval evaluation script for compliance RAG pipeline.

Usage:
    python -m src.evaluation.eval_retrieval

Performs direct hybrid search (bypassing async/LLM pipeline),
and reports Hit@1, Hit@3, Hit@5, and MRR.
"""

from __future__ import annotations

import json
import logging
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


def load_test_queries(filepath: Path) -> list[dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(filepath, encoding="utf-8") as fh:
        queries = json.load(fh)
    logger.info("Loaded %d test queries from %s", len(queries), filepath)
    return queries


def search_single_query(
    query: str,
    embedding_client: BGEM3EmbeddingClient,
    store: MilvusHybridStore,
    top_k: int = 10,
) -> list[str]:
    """Direct hybrid search — returns deduplicated chunk_id list (up to top_k).

    Calls encode() → hybrid_search(), merges dense[:top_k] + sparse[:top_k],
    deduplicates by chunk_id while preserving order.
    """
    embedding = embedding_client.encode(query)
    dense_hits, sparse_hits = store.hybrid_search(
        dense_vector=embedding.dense_vector,
        sparse_vector=embedding.sparse_vector,
        top_k=top_k,
        output_fields=_OUTPUT_FIELDS,
        filters=None,
    )

    seen: set[str] = set()
    chunk_ids: list[str] = []
    for hit in dense_hits[:top_k]:
        cid = str(hit.get("chunk_id", ""))
        if cid and cid not in seen:
            seen.add(cid)
            chunk_ids.append(cid)
    for hit in sparse_hits[:top_k]:
        cid = str(hit.get("chunk_id", ""))
        if cid and cid not in seen:
            seen.add(cid)
            chunk_ids.append(cid)
    return chunk_ids


def evaluate(
    test_queries: list[dict[str, Any]],
    embedding_client: BGEM3EmbeddingClient,
    store: MilvusHybridStore,
) -> list[dict[str, Any]]:
    """Run all test queries through direct hybrid search."""
    results: list[dict[str, Any]] = []
    for i, tq in enumerate(test_queries):
        query = tq.get("query", "")
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(test_queries), query[:80])
        try:
            retrieved_ids = search_single_query(query, embedding_client, store)
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


def print_report(results: list[dict[str, Any]]) -> None:
    """Print evaluation report to stdout."""
    print("\n" + "=" * 60)
    print("  Compliance Retrieval Evaluation Report")
    print("=" * 60)
    print(f"  Total queries evaluated: {len(results)}")

    hit1 = calculate_hit_at_k(results, 1)
    hit3 = calculate_hit_at_k(results, 3)
    hit5 = calculate_hit_at_k(results, 5)
    mrr = calculate_mrr(results)

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
        "Eval complete — Hit@1=%.4f Hit@3=%.4f Hit@5=%.4f MRR=%.4f",
        hit1,
        hit3,
        hit5,
        mrr,
    )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.app.log_level)

    queries_path = Path(__file__).resolve().parent / "test_queries.json"
    if not queries_path.exists():
        logger.error("Test queries file not found: %s", queries_path)
        raise SystemExit(1)

    test_queries = load_test_queries(queries_path)

    store = MilvusHybridStore(settings)
    embedding_client = BGEM3EmbeddingClient(settings.inference.embedding_model_name)

    results = evaluate(test_queries, embedding_client, store)
    print_report(results)


if __name__ == "__main__":
    main()
