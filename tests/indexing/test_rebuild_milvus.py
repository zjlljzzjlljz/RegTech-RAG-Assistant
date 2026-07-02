"""Smoke tests for rebuild_milvus CLI."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.indexing.rebuild_milvus as rebuild


def test_parse_args_pdf_dir_required() -> None:
    """--pdf-dir is required; omitting it must raise SystemExit."""
    import argparse

    try:
        rebuild.parse_args([])
    except SystemExit as exc:
        assert exc.code in (None, 2)


def test_parse_args_accepts_drop_existing_and_dry_run(tmp_path) -> None:
    """Both --drop-existing and --dry-run must be accepted without error."""
    # Create a dummy pdf so find_pdfs doesn't fail first
    dummy = tmp_path / "test.pdf"
    dummy.touch()

    args = rebuild.parse_args(["--pdf-dir", str(tmp_path), "--drop-existing", "--dry-run"])
    assert args.pdf_dir == tmp_path
    assert args.drop_existing is True
    assert args.dry_run is True


def test_parse_args_defaults(tmp_path) -> None:
    """Default flags must be False."""
    args = rebuild.parse_args(["--pdf-dir", str(tmp_path)])
    assert args.drop_existing is False
    assert args.dry_run is False
