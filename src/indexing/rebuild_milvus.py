#!/usr/bin/env python
"""Milvus index rebuild script.

Usage:
    python -m src.indexing.rebuild_milvus --pdf-dir data/raw_pdfs
    python -m src.indexing.rebuild_milvus --pdf-dir data/raw_pdfs --drop-existing

Flags:
    --pdf-dir       Directory containing PDF files to index (required)
    --drop-existing Drop and recreate the collection before ingesting (default: False)
    --dry-run       Parse PDFs and print chunk counts without writing to Milvus
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.settings import configure_logging, get_settings
from src.indexing.milvus_ingest import MilvusIndexer, MilvusHybridStore

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Milvus index from PDFs.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files to ingest.",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the existing collection and recreate before ingesting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse PDFs and count chunks without writing to Milvus.",
    )
    return parser.parse_args(argv)


def find_pdfs(directory: Path) -> list[Path]:
    """Return all .pdf files in *directory* (non-recursive)."""
    if not directory.is_dir():
        raise SystemExit(f"--pdf-dir is not a directory: {directory}")
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {directory}")
    logger.info("Found %d PDF file(s) in %s", len(pdfs), directory)
    return pdfs


def drop_collection(settings) -> None:
    """Drop the Milvus collection if it exists."""
    from pymilvus import utility

    store = MilvusHybridStore(settings)
    store.connect()
    if utility.has_collection(settings.milvus.collection_name, using=settings.milvus.alias):
        utility.drop_collection(settings.milvus.collection_name, using=settings.milvus.alias)
        logger.info("Dropped collection '%s'", settings.milvus.collection_name)
    else:
        logger.info("Collection '%s' does not exist — nothing to drop", settings.milvus.collection_name)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.app.log_level)

    pdfs = find_pdfs(args.pdf_dir)

    if args.drop_existing:
        drop_collection(settings)

    indexer = MilvusIndexer()

    total_chunks = 0
    for pdf_path in pdfs:
        if args.dry_run:
            pages = indexer.parse_pdf(pdf_path)
            chunks = indexer.build_parent_child_chunks(pages)
            logger.info("[DRY-RUN] %s: %d pages → %d chunks", pdf_path.name, len(pages), len(chunks))
            total_chunks += len(chunks)
        else:
            count = indexer.ingest_pdf(pdf_path)
            logger.info("Indexed %s: %d chunks", pdf_path.name, count)
            total_chunks += count

    if args.dry_run:
        print(f"\n[DRY-RUN] Total: {total_chunks} chunks across {len(pdfs)} PDFs")
    else:
        print(f"\nRebuild complete — {total_chunks} chunks indexed across {len(pdfs)} PDFs")


if __name__ == "__main__":
    main()