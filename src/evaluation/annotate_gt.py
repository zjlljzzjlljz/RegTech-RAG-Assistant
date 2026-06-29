#!/usr/bin/env python
"""Auto-annotate ground truth chunk IDs for test queries using Claude.

Usage:
    python -m src.evaluation.annotate_gt

For each query in test_queries.json:
1. Perform direct hybrid search to get top-10 chunks
2. Send query + chunk previews to Claude for relevance judgment
3. Write back ground_truth_chunk_ids to test_queries.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from config.settings import Settings, configure_logging, get_anthropic_client, get_settings
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
    dense_hits, sparse_hits = store.hybrid_search(
        dense_vector=embedding.dense_vector,
        sparse_vector=embedding.sparse_vector,
        top_k=top_k,
        output_fields=_OUTPUT_FIELDS,
        filters=None,
    )

    seen: set[str] = set()
    chunks: list[dict[str, Any]] = []
    for hit in [*dense_hits[:top_k], *sparse_hits[:top_k]]:
        cid = str(hit.get("chunk_id", ""))
        if cid and cid not in seen:
            seen.add(cid)
            chunks.append(
                {
                    "chunk_id": cid,
                    "source_file": str(hit.get("source_file", "")),
                    "page_number": hit.get("page_number"),
                    "text": str(hit.get("text", ""))[:200],
                }
            )
        if len(chunks) >= top_k:
            break
    return chunks


def build_annotation_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """Build prompt for Claude to judge chunk relevance."""
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

    return (
        "You are a compliance retrieval evaluator. For the given query, identify "
        "which chunks from the search results are RELEVANT to answering the query. "
        "A chunk is relevant if it contains regulatory guidance, AML/KYC rules, "
        "or compliance procedures that directly address the question.\n\n"
        f"## Query\n{query}\n\n"
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
) -> list[dict[str, Any]]:
    """Annotate each query by retrieving top-K and asking Claude for relevance."""
    client = get_anthropic_client(settings)
    annotated_count = 0

    for i, tq in enumerate(test_queries):
        query = tq.get("query", "")
        # Skip if already annotated
        if tq.get("ground_truth_chunk_ids"):
            logger.info("[%d/%d] Skipping (already annotated): %s", i + 1, len(test_queries), query[:60])
            continue

        logger.info("[%d/%d] Annotating: %s", i + 1, len(test_queries), query[:80])
        try:
            chunks = search_chunks(query, embedding_client, store)
            if not chunks:
                logger.warning("No chunks retrieved for query: %s", query[:60])
                continue

            prompt = build_annotation_prompt(query, chunks)
            response = client.messages.create(
                model=settings.llm.model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": "You are a compliance retrieval evaluator. Return valid JSON only.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},
            )

            body = ""
            for block in response.content:
                block_text = getattr(block, "text", None)
                if block_text:
                    body += str(block_text)

            try:
                parsed = json.loads(body)
                relevant_ids = parsed.get("relevant_ids", [])
            except json.JSONDecodeError:
                logger.warning("Non-JSON response for query: %s... using empty list", query[:40])
                relevant_ids = []

            tq["ground_truth_chunk_ids"] = relevant_ids if isinstance(relevant_ids, list) else []
            annotated_count += 1
            logger.info("  → %d relevant chunks identified", len(tq["ground_truth_chunk_ids"]))

        except Exception as exc:  # noqa: BLE001
            logger.error("Annotation failed for query '%s': %s", query[:60], exc)
            tq["ground_truth_chunk_ids"] = []

        # Rate-limit friendly
        time.sleep(0.5)

    logger.info("Annotated %d new queries", annotated_count)
    return test_queries


def main() -> None:
    settings = get_settings()
    configure_logging(settings.app.log_level)

    if not _QUERIES_FILE.exists():
        logger.error("Test queries file not found: %s", _QUERIES_FILE)
        raise SystemExit(1)

    test_queries = load_test_queries(_QUERIES_FILE)
    logger.info("Loaded %d test queries", len(test_queries))

    store = MilvusHybridStore(settings)
    embedding_client = BGEM3EmbeddingClient(settings.inference.embedding_model_name)

    test_queries = annotate_queries(test_queries, embedding_client, store, settings)
    save_test_queries(_QUERIES_FILE, test_queries)

    print(f"\nAnnotation complete. {sum(1 for q in test_queries if q['ground_truth_chunk_ids'])}/{len(test_queries)} queries have relevant chunks.")


if __name__ == "__main__":
    main()
