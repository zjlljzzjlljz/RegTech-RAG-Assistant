from __future__ import annotations

from src.evaluation.eval_planned_retrieval import _percentile, _stage_metrics, parse_args


def test_parse_args_defaults_to_all_v2_suite() -> None:
    args = parse_args([])
    assert args.suite == "all"
    assert args.limit == 0
    assert args.no_reranker is False


def test_stage_metrics_are_split_by_language_suite() -> None:
    records = [
        {
            "suite": "en",
            "ground_truth_chunk_ids": ["child-1"],
            "ids": ["child-1"],
        },
        {
            "suite": "zh",
            "ground_truth_chunk_ids": ["child-2"],
            "ids": ["other"],
        },
    ]
    metrics = _stage_metrics(records, "ids")
    assert metrics["overall"]["hit_at_1"] == 0.5
    assert metrics["by_suite"]["en"]["hit_at_1"] == 1.0
    assert metrics["by_suite"]["zh"]["hit_at_1"] == 0.0


def test_percentile_uses_ordered_nearest_rank() -> None:
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 3.0
