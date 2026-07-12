#!/usr/bin/env python
"""Auto-annotate ground truth chunk IDs with the configured judge model.

Usage:
    python -m src.evaluation.annotate_gt --force --output-file test_queries_v2.json

For each query in test_queries.json:
1. Perform direct hybrid search to get top-10 chunks
2. Send query + chunk previews to the configured judge for relevance judgment
3. Write v2 ground_truth_chunk_ids to a separate JSON file
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from config.settings import Settings, configure_logging, get_settings
from src.inference import create_llm_client
from src.inference.llm_client import parse_json_object
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
_QUERIES_FILE = Path(__file__).resolve().parent / "test_queries.json"
_RRF_K = 60
_JUDGE_TEXT_LIMIT = 1200


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate retrieval ground truth for the active collection.")
    parser.add_argument("--queries-file", type=Path, default=_QUERIES_FILE)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace existing ground-truth chunk IDs.")
    parser.add_argument("--limit", type=int, default=0, help="Annotate at most N queries; 0 means all.")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def load_test_queries(filepath: Path) -> list[dict[str, Any]]:
    with open(filepath, encoding="utf-8") as fh:
        return json.load(fh)


def save_test_queries(filepath: Path, queries: list[dict[str, Any]]) -> None:
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(queries, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d queries to %s", len(queries), filepath)


def search_chunks(
    query: str,
    embedding_client: BGEM3EmbeddingClient,
    store: MilvusHybridStore,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Direct hybrid search returning top-K chunk dicts with deduplication."""
    embedding = embedding_client.encode(
        query,
        prompt="Represent this sentence for searching relevant passages: ",
    )
    search_k = max(top_k * 3, 20)
    dense_hits, sparse_hits = store.hybrid_search(
        dense_vector=embedding.dense_vector,
        sparse_vector=embedding.sparse_vector,
        top_k=search_k,
        output_fields=_OUTPUT_FIELDS,
        filters='chunk_type == "child"',
    )

    scored: dict[str, float] = {}
    canonical: dict[str, dict[str, Any]] = {}
    for result_set in (dense_hits, sparse_hits):
        for rank, hit in enumerate(result_set[:search_k], start=1):
            chunk_id = str(hit.get("chunk_id", ""))
            if not chunk_id:
                continue
            canonical.setdefault(chunk_id, hit)
            scored[chunk_id] = scored.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)

    fused_ids = sorted(scored, key=lambda chunk_id: (-scored[chunk_id], chunk_id))[:top_k]
    chunks: list[dict[str, Any]] = []
    for chunk_id in fused_ids:
        hit = canonical[chunk_id]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source_file": str(hit.get("source_file", "")),
                "page_number": hit.get("page_number"),
                "text": str(hit.get("text", ""))[:_JUDGE_TEXT_LIMIT],
            }
        )
    return chunks


def build_annotation_prompt(
    query: str,
    chunks: list[dict[str, Any]],
    reference_answer: str = "",
) -> str:
    """Build a provider-neutral prompt for the configured judge model."""
    chunk_lines: list[str] = []
    for c in chunks:
        loc = (
            f"{c['source_file']}#page-{c['page_number']}"
            if c.get("page_number")
            else c["source_file"]
        )
        chunk_lines.append(
            f"[ID: {c['chunk_id']}] [{loc}]\n{c['text']}\n"
        )

    reference_section = (
        f"## Human Reference Answer\n{reference_answer}\n\n"
        if reference_answer
        else ""
    )
    return (
        "You are a compliance retrieval evaluator. For the given query, identify "
        "the smallest set of chunks that directly supports the answer. Select no "
        "more than 3 chunks. Reject chunks that only mention the same broad topic, "
        "contain incomplete fragments, or do not provide evidence for a material "
        "part of the reference answer.\n\n"
        f"## Query\n{query}\n\n"
        f"{reference_section}"
        f"## Candidate Chunks\n\n{''.join(chunk_lines)}\n"
        "Return a JSON object with key 'relevant_ids' containing a list of "
        "chunk IDs that are relevant. Return an empty list if none are relevant.\n"
        'Example: {"relevant_ids": ["chunk-uuid-1", "chunk-uuid-3"]}'
    )


def annotate_queries(
    test_queries: list[dict[str, Any]],
    embedding_client: BGEM3EmbeddingClient,
    store: MilvusHybridStore,
    settings: Settings,
    *,
    force: bool = False,
    limit: int = 0,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Annotate each query by retrieving top-K and asking the configured judge for relevance."""
    client = create_llm_client(settings.llm_roles.judge, settings)
    annotated_count = 0

    if force:
        for query_record in test_queries:
            query_record["ground_truth_chunk_ids"] = []

    processed = 0
    for i, tq in enumerate(test_queries):
        query = tq.get("query", "")
        # Skip if already annotated
        if tq.get("ground_truth_chunk_ids"):
            logger.info("[%d/%d] Skipping (already annotated): %s", i + 1, len(test_queries), query[:60])
            continue
        if limit > 0 and processed >= limit:
            continue

        logger.info("[%d/%d] Annotating: %s", i + 1, len(test_queries), query[:80])
        processed += 1
        try:
            reference_answer = str(tq.get("ground_truth_answer", "")).strip()
            retrieval_query = f"{query}\n{reference_answer}" if reference_answer else query
            chunks = search_chunks(retrieval_query, embedding_client, store, top_k=top_k)
            if not chunks:
                logger.warning("No chunks retrieved for query: %s", query[:60])
                continue

            prompt = build_annotation_prompt(
                query,
                chunks,
                reference_answer,
            )
            response = client.complete(
                system="You are a compliance retrieval evaluator. Return valid JSON only.",
                prompt=prompt,
                json_mode=True,
            )
            try:
                parsed = parse_json_object(response.text)
                relevant_ids = parsed.get("relevant_ids", [])
            except Exception:
                logger.warning("Non-JSON response for query: %s... using empty list", query[:40])
                relevant_ids = []

            candidate_ids = {chunk["chunk_id"] for chunk in chunks}
            tq["ground_truth_chunk_ids"] = (
                [str(chunk_id) for chunk_id in relevant_ids if str(chunk_id) in candidate_ids]
                if isinstance(relevant_ids, list)
                else []
            )
            annotated_count += 1
            logger.info("  → %d relevant chunks identified", len(tq["ground_truth_chunk_ids"]))

        except Exception as exc:  # noqa: BLE001
            logger.error("Annotation failed for query '%s': %s", query[:60], exc)
            tq["ground_truth_chunk_ids"] = []

        # Rate-limit friendly
        time.sleep(0.5)

    logger.info("Annotated %d new queries", annotated_count)
    return test_queries


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.app.log_level)

    if not args.queries_file.exists():
        logger.error("Test queries file not found: %s", args.queries_file)
        raise SystemExit(1)

    all_test_queries = load_test_queries(args.queries_file)
    test_queries = all_test_queries[: args.limit] if args.limit > 0 else all_test_queries
    logger.info("Loaded %d test queries (%d selected)", len(all_test_queries), len(test_queries))

    store = MilvusHybridStore(settings)
    embedding_client = BGEM3EmbeddingClient(settings.inference.embedding_model_name)

    test_queries = annotate_queries(
        test_queries,
        embedding_client,
        store,
        settings,
        force=args.force,
        limit=0,
        top_k=args.top_k,
    )
    output_file = args.output_file or args.queries_file
    save_test_queries(output_file, test_queries)

    print(f"\nAnnotation complete. {sum(1 for q in test_queries if q['ground_truth_chunk_ids'])}/{len(test_queries)} queries have relevant chunks.")


if __name__ == "__main__":
    main()
