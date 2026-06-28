from __future__ import annotations

import json
import logging
import time
from typing import Any

import streamlit as st

from config.settings import configure_logging, get_settings
from src.agent import AuditResult, ComplianceAgentGraph
from src.indexing import BGEM3EmbeddingClient, MilvusHybridStore
from src.retrieval import ComplianceRetrievalPipeline, CrossEncoderRerankerClient
from src.storage import TransactionLog, TransactionRepository

LOGGER = logging.getLogger(__name__)
_DEFAULT_QUERY = "What are the key AML/KYC compliance requirements for cross-border transactions in Hong Kong?"


@st.cache_resource(show_spinner=False)
def get_agent_graph() -> ComplianceAgentGraph:
    settings = get_settings()
    import threading as _threading

    # BGE-M3 + Milvus load fast now (models cached)
    store = MilvusHybridStore(settings)
    embedding_client = BGEM3EmbeddingClient(settings.inference.embedding_model_name)

    # Reranker may still be downloading — try with timeout
    _reranker_result: dict[str, Any] = {}

    def _load_reranker() -> None:
        try:
            _reranker_result["reranker"] = CrossEncoderRerankerClient(settings.inference.reranker_model_name)
        except Exception:  # noqa: BLE001
            pass

    _thread = _threading.Thread(target=_load_reranker, daemon=True)
    _thread.start()
    _thread.join(timeout=30)

    reranker = _reranker_result.get("reranker")
    if reranker is None:
        if _thread.is_alive():
            LOGGER.warning("Reranker model downloading — using dense-only mode for now")
        else:
            LOGGER.warning("Reranker unavailable — using dense-only mode")

    pipeline = ComplianceRetrievalPipeline(
        store=store,
        embedding_client=embedding_client,
        reranker=reranker,
        settings=settings,
    )
    return ComplianceAgentGraph(
        retrieval_pipeline=pipeline,
        settings=settings,
    )


def _ensure_singletons() -> ComplianceAgentGraph:
    if "compliance_agent_graph" not in st.session_state:
        st.session_state.compliance_agent_graph = get_agent_graph()
    if "transaction_repository" not in st.session_state:
        try:
            st.session_state.transaction_repository = TransactionRepository()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Transaction repository unavailable: %s", exc)
            st.session_state.transaction_repository = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_log_id" not in st.session_state:
        st.session_state.last_log_id = None
    if "last_feedback_value" not in st.session_state:
        st.session_state.last_feedback_value = None
    return st.session_state.compliance_agent_graph


def _render_token_metrics(token_metrics: dict[str, int]) -> None:
    st.caption("Token Usage")
    if not token_metrics:
        st.write("No token usage recorded yet.")
        return
    left, middle, right = st.columns(3)
    left.metric("Prompt", token_metrics.get("prompt_tokens", 0))
    middle.metric("Completion", token_metrics.get("completion_tokens", 0))
    right.metric("Total", token_metrics.get("total_tokens", 0))


def _render_result(result: AuditResult) -> None:
    if result.error:
        st.error(result.error)

    st.subheader("Compliance Answer")
    if result.answer:
        st.markdown(result.answer)
    else:
        st.info("No answer was generated.")

    st.subheader("Verified Claims")
    if result.claims:
        for i, claim in enumerate(result.claims, 1):
            confidence = claim.get("confidence", "medium")
            uncertain = " ⚠️ Uncertain" if claim.get("uncertain") else ""
            icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            sources = ", ".join(claim.get("source_ids", []))
            st.markdown(f"{i}. {icon} **[{confidence.upper()}{uncertain}]** {claim.get('statement', '')}")
            st.caption(f"   Sources: `{sources}`")
    else:
        st.write("No claims extracted.")

    st.subheader("Cited Sources")
    if result.cite_sources:
        for source in result.cite_sources:
            st.markdown(f"- `{source}`")
    else:
        st.write("No cited sources available.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Claims Count", len(result.claims))
    col2.metric("Sources Count", len(result.cite_sources))

    st.subheader("Token Metrics")
    st.json(result.token_metrics or {})

    feedback_value = st.feedback("thumbs", key="result_feedback")
    _persist_feedback(feedback_value)


def _normalize_feedback_value(feedback_value: int | None) -> str | None:
    if feedback_value is None:
        return None
    if feedback_value in (1, +1):
        return "positive"
    if feedback_value in (0, -1):
        return "negative"
    return None


def _persist_feedback(feedback_value: int | None) -> None:
    feedback = _normalize_feedback_value(feedback_value)
    if feedback is None:
        return

    log_id = st.session_state.get("last_log_id")
    repo = st.session_state.get("transaction_repository")
    feedback_state = st.session_state.get("last_feedback_value")
    if log_id is None or repo is None or feedback_state == feedback:
        return

    try:
        repo.update_feedback(int(log_id), feedback)
        st.session_state.last_feedback_value = feedback
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feedback persistence failed: %s", exc)


def _run_audit(query: str) -> AuditResult:
    agent_graph = _ensure_singletons()
    start = time.perf_counter()
    initial_state: dict[str, Any] = {
        "user_query": query,
    }
    result = agent_graph.invoke(initial_state)
    latency_ms = (time.perf_counter() - start) * 1000

    try:
        repo = st.session_state.get("transaction_repository")
        if repo is None:
            repo = TransactionRepository()
            st.session_state.transaction_repository = repo

        log = TransactionLog(
            query=query,
            answer=result.answer,
            claims_json=json.dumps(result.claims, ensure_ascii=False),
            error_message=result.error or None,
            prompt_tokens=int(result.token_metrics.get("prompt_tokens", 0)),
            completion_tokens=int(result.token_metrics.get("completion_tokens", 0)),
            total_tokens=int(result.token_metrics.get("total_tokens", 0)),
            iterations=int(getattr(result, "iterations", 0) or 0),
            latency_ms=latency_ms,
        )
        st.session_state.last_log_id = repo.insert(log)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Transaction persistence failed: %s", exc)
        st.session_state.last_log_id = None

    return result


def main() -> None:
    settings = get_settings()
    configure_logging(settings.app.log_level)

    st.set_page_config(
        page_title=settings.app.page_title,
        page_icon=settings.app.page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(settings.app.page_title)
    st.caption("Compliance Q&A workspace powered by Milvus hybrid retrieval and LangGraph orchestration.")

    with st.sidebar:
        st.header("Controls")
        query = st.text_area(
            "Compliance query",
            value=st.session_state.get("last_query", _DEFAULT_QUERY),
            height=160,
            help="Ask a compliance question about AML, KYC, cross-border regulations, etc.",
        )
        run_clicked = st.button("Run Query", type="primary", use_container_width=True)
        st.divider()
        _render_token_metrics(
            getattr(st.session_state.get("last_result"), "token_metrics", {})
            if st.session_state.get("last_result")
            else {}
        )

    if run_clicked:
        if not query.strip():
            st.error("Please enter a compliance query before running.")
        else:
            st.session_state.last_query = query
            st.session_state.last_feedback_value = None
            st.session_state.last_log_id = None
            if "result_feedback" in st.session_state:
                del st.session_state["result_feedback"]
            try:
                with st.spinner("Running compliance analysis..."):
                    result = _run_audit(query.strip())
                st.session_state.last_result = result
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Compliance workflow failed")
                st.error(f"Compliance workflow failed: {exc}")
                return

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Enter a compliance query from the sidebar to get started.")
        return

    _render_result(result)


if __name__ == "__main__":
    main()
