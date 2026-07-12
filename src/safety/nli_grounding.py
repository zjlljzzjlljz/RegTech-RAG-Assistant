from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from config.settings import Settings, get_settings


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")


@dataclass(frozen=True)
class GroundingResult:
    passed: bool
    unsupported_sentences: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class NLIGroundingVerifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pipeline: Any = None

    def verify(self, answer: str, chunks: list[Any]) -> GroundingResult:
        sentences = self._factual_sentences(answer)
        if not sentences:
            return GroundingResult(passed=True)
        evidence = [str(getattr(chunk, "text", "")).strip() for chunk in chunks]
        evidence = [text for text in evidence if text]
        if not evidence:
            return GroundingResult(passed=False, unsupported_sentences=sentences)

        scores: dict[str, float] = {}
        unsupported: list[str] = []
        for sentence in sentences:
            score = max(self._entailment_score(premise, sentence) for premise in evidence)
            scores[sentence] = score
            if score < self.settings.inference.nli_entailment_threshold:
                unsupported.append(sentence)
        return GroundingResult(passed=not unsupported, unsupported_sentences=unsupported, scores=scores)

    def _entailment_score(self, premise: str, hypothesis: str) -> float:
        service_url = self.settings.inference.nli_service_url
        if service_url:
            response = httpx.post(
                f"{service_url.rstrip('/')}/nli",
                json={"premise": premise, "hypothesis": hypothesis},
                timeout=self.settings.inference.request_timeout_seconds,
            )
            response.raise_for_status()
            return float(response.json()["entailment"])

        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.settings.inference.nli_model_name,
                top_k=None,
                truncation=True,
            )
        result = self._pipeline({"text": premise, "text_pair": hypothesis})
        labels = result[0] if result and isinstance(result[0], list) else result
        for item in labels:
            label = str(item.get("label", "")).lower()
            if "entail" in label or label in {"label_2", "2"}:
                return float(item.get("score", 0.0))
        return 0.0

    def _factual_sentences(self, answer: str) -> list[str]:
        output: list[str] = []
        for part in _SENTENCE_SPLIT.split(answer):
            sentence = re.sub(r"^[#>*\-\d.\s]+", "", part).strip()
            if len(sentence) < 20:
                continue
            if sentence.endswith(":") or sentence.endswith("："):
                continue
            output.append(sentence)
        return output


__all__ = ["GroundingResult", "NLIGroundingVerifier"]
