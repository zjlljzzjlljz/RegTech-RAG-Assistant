from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from config.settings import get_settings
from src.agent.graph_agent import ComplianceAgentGraph
from src.inference import LLMResponse
from src.retrieval.query_pipeline import AuditDecision, AuditIntent, RetrievedChunk, RetrievalResult


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, **kwargs):
        return LLMResponse(text=self.text)


class PassingGroundingVerifier:
    def verify(self, answer, chunks):
        return SimpleNamespace(passed=True, unsupported_sentences=[], scores={})


class FakePipeline:
    async def retrieve(self, request):
        chunk = RetrievedChunk(
            chunk_id="child-1",
            parent_id="parent-1",
            text="Institutions must conduct customer due diligence.",
            source_file="guide.pdf",
            page_number=1,
            chunk_type="parent_context",
            score=0.9,
        )
        return RetrievalResult(
            audit_decision=AuditDecision(AuditIntent.COMPLIANCE_QA, "test"),
            expanded_queries=[request.user_query],
            hyde_hypothesis="",
            retrieved_chunks=[chunk],
            reranked_chunks=[chunk],
            answer_context=chunk.text,
        )


def build_graph(auditor_text: str) -> ComplianceAgentGraph:
    settings = get_settings()
    settings = replace(settings, inference=replace(settings.inference, nli_enabled=False))
    return ComplianceAgentGraph(
        settings=settings,
        draft_client=FakeClient('{"answer_summary":"ok","claims":[]}'),
        auditor_client=FakeClient(auditor_text),
        grounding_verifier=PassingGroundingVerifier(),
    )


def test_auditor_invalid_json_is_fail_closed() -> None:
    graph = build_graph("APPROVED")
    result = graph.auditor_review_node(
        {
            "user_query": "What is CDD?",
            "draft_answer": "CDD is required for regulated institutions.",
            "retrieved_chunks": [],
            "current_iteration": 0,
        }
    )
    assert result["audit_status"] == "error"
    assert result["error_message"]


def test_rejected_draft_is_blocked_from_final_output() -> None:
    graph = build_graph('{"approved":false,"feedback":"unsupported"}')
    result = graph.finalize_node(
        {
            "draft_answer": "An unsupported compliance statement.",
            "claims": [],
            "retrieved_chunks": [],
            "audit_status": "max_iterations",
        }
    )
    assert "本报告未经审计通过" in result["final_output"]
    assert "unsupported compliance statement" not in result["final_output"]


def test_max_iterations_block_draft_in_full_graph() -> None:
    settings = get_settings()
    settings = replace(settings, inference=replace(settings.inference, nli_enabled=False))
    graph = ComplianceAgentGraph(
        settings=settings,
        retrieval_pipeline=FakePipeline(),
        draft_client=FakeClient(
            '{"answer_summary":"CDD is required.","claims":'
            '[{"statement":"CDD is required.","source_ids":["chunk-1"],"confidence":"high"}]}'
        ),
        auditor_client=FakeClient('{"approved":false,"feedback":"Revise grounding."}'),
        grounding_verifier=PassingGroundingVerifier(),
    )
    result = graph.invoke({"user_query": "What is required?"})
    assert result.audit_status == "max_iterations"
    assert result.iterations == settings.retrieval.max_audit_iterations
    assert "本报告未经审计通过" in result.answer
    assert result.claims == []
