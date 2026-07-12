from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_LIMITS = {
    "recall_at_10": (0.02, None),
    "mrr": (0.03, None),
    "citation_precision": (None, 0.95),
    "unsupported_sentence_rate": (None, None),
    "unsafe_auditor_pass_rate": (None, 0.0),
}


def evaluate_gate(baseline: dict[str, float], candidate: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric, (max_drop, minimum) in DEFAULT_LIMITS.items():
        if metric not in candidate:
            failures.append(f"Missing candidate metric: {metric}")
            continue
        value = float(candidate[metric])
        if minimum is not None and value < minimum:
            failures.append(f"{metric}={value:.4f} is below minimum {minimum:.4f}")
        if max_drop is not None and metric in baseline:
            drop = float(baseline[metric]) - value
            if drop - max_drop > 1e-9:
                failures.append(f"{metric} regressed by {drop:.4f}; maximum allowed is {max_drop:.4f}")
    if float(candidate.get("unsupported_sentence_rate", 1.0)) > 0.02:
        failures.append("unsupported_sentence_rate exceeds 0.02")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    candidate = candidate.get("scores", candidate)
    baseline = baseline.get("scores", baseline)
    failures = evaluate_gate(baseline, candidate)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Evaluation regression gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
