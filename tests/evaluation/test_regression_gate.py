from src.evaluation.regression_gate import evaluate_gate


def test_regression_gate_passes_at_thresholds() -> None:
    baseline = {"recall_at_10": 0.8, "mrr": 0.7}
    candidate = {
        "recall_at_10": 0.78,
        "mrr": 0.67,
        "citation_precision": 0.95,
        "unsupported_sentence_rate": 0.02,
        "unsafe_auditor_pass_rate": 0.0,
    }
    assert evaluate_gate(baseline, candidate) == []


def test_regression_gate_rejects_unsupported_sentences() -> None:
    candidate = {
        "recall_at_10": 1.0,
        "mrr": 1.0,
        "citation_precision": 1.0,
        "unsupported_sentence_rate": 0.03,
        "unsafe_auditor_pass_rate": 0.0,
    }
    assert any("unsupported_sentence_rate" in failure for failure in evaluate_gate({}, candidate))
