from __future__ import annotations

import asyncio
from dataclasses import replace

from config.settings import get_settings
from src.indexing.milvus_ingest import BGEEmbeddingResult
from src.retrieval.query_pipeline import AuditDecision, AuditIntent, ComplianceRetrievalPipeline, RetrievalRequest, RetrievedChunk


class FakePlanner:
    def __init__(self) -> None:
        self.hyde_calls = 0

    def classify_intent(self, query, history=None):
        return AuditDecision(AuditIntent.COMPLIANCE_QA, "test")

    def generate_multi_queries(self, query):
        return [query, f"{query} rewrite"]

    def generate_hyde_document(self, query):
        self.hyde_calls += 1
        return f"hypothetical {query}"


class FakeEncoder:
    def encode(self, text, prompt=None):
        return BGEEmbeddingResult([1.0, 0.0], {1: 1.0})


class FakeStore:
    def hybrid_search(self, **kwargs):
        hit = {
            "chunk_id": "child-1",
            "parent_id": "parent-1",
            "chunk_type": "child",
            "source_file": "guide.pdf",
            "page_number": 1,
            "text": "CDD evidence",
            "metadata": {},
            "score": 1.0,
        }
        return [hit], [hit]


class FakeReranker:
    def __init__(self, score: float) -> None:
        self.score = score

    def rerank(self, query, chunks, top_k, score_threshold=0.0):
        chunk = chunks[0]
        return [RetrievedChunk(**{**chunk.__dict__, "score": self.score})]


def build_pipeline(score: float):
    settings = get_settings()
    retrieval = replace(
        settings.retrieval,
        parent_backfill=False,
        rerank_score_threshold=0.0,
        hyde_score_threshold=0.25,
        hyde_margin_threshold=0.05,
    )
    planner = FakePlanner()
    pipeline = ComplianceRetrievalPipeline(
        store=FakeStore(),
        embedding_client=FakeEncoder(),
        reranker=FakeReranker(score),
        llm_provider=planner,
        settings=replace(settings, retrieval=retrieval),
    )
    return pipeline, planner


def test_hyde_is_skipped_for_confident_retrieval() -> None:
    pipeline, planner = build_pipeline(0.9)
    result = asyncio.run(pipeline.retrieve(RetrievalRequest("CDD")))
    assert result.hyde_hypothesis == ""
    assert planner.hyde_calls == 0


def test_hyde_runs_for_low_confidence_retrieval() -> None:
    pipeline, planner = build_pipeline(0.1)
    result = asyncio.run(pipeline.retrieve(RetrievalRequest("CDD")))
    assert result.hyde_hypothesis.startswith("hypothetical")
    assert planner.hyde_calls == 1


def test_query_expansion_is_gated_by_language_mix() -> None:
    pipeline, _ = build_pipeline(0.9)

    assert pipeline._should_expand_query("客户尽职审查有什么要求？") is True
    assert pipeline._should_expand_query("What are the CDD requirements?") is False
    assert pipeline._should_expand_query(
        "authorized institution 什么时候做 CDD including occasional transaction threshold"
    ) is False


def test_hyde_is_disabled_without_reranker_scores() -> None:
    pipeline, _ = build_pipeline(0.1)
    pipeline.reranker = None

    assert pipeline._should_trigger_hyde([]) is False
