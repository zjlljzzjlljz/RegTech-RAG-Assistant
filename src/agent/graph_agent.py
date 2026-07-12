from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from config.settings import Settings, get_settings
from src.inference import LLMClient, create_llm_client
from src.inference.llm_client import LLMError, parse_json_object
from src.retrieval.query_pipeline import (
    ComplianceRetrievalPipeline,
    RetrievedChunk,
    RetrievalRequest,
    RetrievalResult,
)
from src.safety import NLIGroundingVerifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State & result schemas
# ---------------------------------------------------------------------------


class AuditState(TypedDict, total=False):
    user_query: str
    audit_intent: str  # fixed to "compliance_qa"
    retrieval_result: Optional[Any]  # RetrievalResult | None
    retrieved_chunks: list[Any]  # list[RetrievedChunk]
    current_iteration: int  # 1-based iteration counter
    draft_answer: str  # Draftee output (approved draft or revision target)
    audit_feedback: str  # Auditor feedback (empty = approved)
    audit_status: str  # pending | approved | rejected | error | max_iterations
    claims: list[dict[str, Any]]  # validated claims with source_ids
    cite_sources: list[str]  # human-readable citation labels
    error_message: str
    final_output: str
    token_metrics: dict[str, int]
    grounding_scores: dict[str, float]


@dataclass
class AuditResult:
    answer: str
    claims: list[dict[str, Any]]
    cite_sources: list[str]
    token_metrics: dict[str, int]
    error: Optional[str]
    retrieved_chunks: list[RetrievedChunk] | None = None
    iterations: int = 0
    audit_status: str = "pending"
    grounding_scores: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Anti-hallucination helpers
# ---------------------------------------------------------------------------

_INSUFFICIENT_EVIDENCE_MSG = (
    "# Compliance Q&A — Insufficient Evidence\n\n"
    "当前文档库无法支撑此问题的回答。"
    "The available regulatory documents do not contain sufficient information "
    "to answer this question reliably. Please rephrase your query or consult "
    "a compliance officer for this specific inquiry."
)

_UNAUDITED_REPORT_MSG = (
    "# 本报告未经审计通过\n\n"
    "The generated draft was blocked because it did not pass the compliance audit "
    "within the configured iteration limit. Consult a compliance officer and review "
    "the internal audit trail before using any draft content."
)


def _build_chunk_id_set(chunks: list[Any]) -> set[str]:
    """Extract all chunk IDs from retrieved chunks for source verification."""
    ids: set[str] = set()
    for c in chunks:
        cid = getattr(c, "chunk_id", None) or ""
        if cid:
            ids.add(str(cid))
    return ids


def _build_chunk_cite_map(chunks: list[Any]) -> dict[str, str]:
    """Map chunk_id → human-readable citation label."""
    cite_map: dict[str, str] = {}
    for c in chunks:
        cid = str(getattr(c, "chunk_id", "") or "")
        if not cid:
            continue
        loc = getattr(c, "source_file", "unknown")
        pn = getattr(c, "page_number", None)
        if pn is not None:
            loc = f"{loc}#page-{pn}"
        cite_map[cid] = loc
    return cite_map


def _validate_claims(
    claims: list[dict[str, Any]],
    valid_chunk_ids: set[str],
    cite_map: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate claims against retrieved chunks.

    Returns (validated_claims, cite_sources).
    - Strips claims with empty source_ids or no matching valid chunk.
    - Marks low-confidence claims as "uncertain".
    - Accumulates unique cite_sources from verified claims.
    """
    validated: list[dict[str, Any]] = []
    cite_sources: list[str] = []

    for claim in claims:
        source_ids: list[str] = []
        raw_ids = claim.get("source_ids", [])
        if isinstance(raw_ids, list):
            source_ids = [str(sid) for sid in raw_ids if str(sid).strip()]

        # Layer 4: source_ids must not be empty; must map to real chunks
        if not source_ids:
            logger.warning("Claim stripped — empty source_ids: %s", claim.get("statement", "")[:80])
            continue

        matched = [sid for sid in source_ids if sid in valid_chunk_ids]
        if not matched:
            logger.warning("Claim stripped — no source_ids found in retrieved chunks: %s", source_ids)
            continue

        # Mark low-confidence claims
        confidence = str(claim.get("confidence", "medium")).lower()
        if confidence == "low":
            claim["uncertain"] = True

        claim["source_ids"] = matched
        validated.append(claim)

        for sid in matched:
            cite_label = cite_map.get(sid, sid)
            if cite_label not in cite_sources:
                cite_sources.append(cite_label)

    return validated, cite_sources


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


class ComplianceAgentGraph:
    """LangGraph-powered compliance Q&A with Draft-Audit feedback loop (max 3 iterations)."""

    def __init__(
        self,
        retrieval_pipeline: ComplianceRetrievalPipeline | None = None,
        settings: Settings | None = None,
        draft_client: LLMClient | None = None,
        auditor_client: LLMClient | None = None,
        grounding_verifier: NLIGroundingVerifier | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pipeline = retrieval_pipeline
        self._draft_client = draft_client or create_llm_client(self._settings.llm_roles.draft, self._settings)
        self._auditor_client = auditor_client or create_llm_client(self._settings.llm_roles.auditor, self._settings)
        self._grounding_verifier = grounding_verifier or NLIGroundingVerifier(self._settings)
        self._compiled: Any = None

    @property
    def _max_iterations(self) -> int:
        return self._settings.retrieval.max_audit_iterations

    # ------------------------------------------------------------------
    # Helper: invoke role-specific model
    # ------------------------------------------------------------------

    def _llm_request(self, client: LLMClient, prompt: str, system_text: str) -> Any:
        return client.complete(system=system_text, prompt=prompt, json_mode=True)

    @staticmethod
    def _usage_tokens(response: Any) -> dict[str, int]:
        return {
            "prompt_tokens": int(getattr(response, "prompt_tokens", 0)),
            "completion_tokens": int(getattr(response, "completion_tokens", 0)),
            "total_tokens": int(getattr(response, "total_tokens", 0)),
        }

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any] | None:
        try:
            return parse_json_object(text)
        except LLMError:
            return None

    @staticmethod
    def _extract_text(response: Any) -> str:
        return str(getattr(response, "text", "")).strip()

    # ------------------------------------------------------------------
    # Node 1 – retrieve (Draftee-Auditor loop entry point)
    # ------------------------------------------------------------------

    def retrieve_node(self, state: AuditState) -> dict[str, Any]:
        try:
            start = time.perf_counter()
            query = str(state.get("user_query", ""))
            chunks: list[RetrievedChunk] = []
            retrieval_result: RetrievalResult | None = None

            if self._pipeline is not None:
                import asyncio as _asyncio
                import nest_asyncio as _nest

                try:
                    loop = _asyncio.get_event_loop()
                except RuntimeError:
                    loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(loop)
                _nest.apply(loop)
                retrieval_result = loop.run_until_complete(
                    self._pipeline.retrieve(RetrievalRequest(user_query=query))
                )
                chunks = retrieval_result.reranked_chunks
                logger.info(
                    "retrieve_node → pipeline returned %d reranked chunks (from %d raw, %d fused)",
                    len(retrieval_result.reranked_chunks),
                    len(retrieval_result.retrieved_chunks),
                    len(retrieval_result.expanded_queries),
                )
            else:
                # Pipeline unavailable — return empty chunks, let downstream handle insufficient evidence
                logger.warning("retrieve_node → pipeline unavailable, returning empty chunks")
                return {
                    "retrieval_result": None,
                    "retrieved_chunks": [],
                    "error_message": "",
                }

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("retrieve_node → %d chunks, latency=%.1f ms", len(chunks), elapsed_ms)
            return {
                "retrieval_result": retrieval_result,
                "retrieved_chunks": chunks,
                "error_message": "",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("retrieve_node failed: %s", exc)
            return {
                "retrieval_result": None,
                "retrieved_chunks": [],
                "error_message": f"Retrieval failed: {exc}",
            }

    # ------------------------------------------------------------------
    # Node 2 – generate_draft (Draftee role)
    # ------------------------------------------------------------------

    def generate_draft_node(self, state: AuditState) -> dict[str, Any]:
        try:
            start = time.perf_counter()
            query = str(state.get("user_query", ""))
            chunks = list(state.get("retrieved_chunks") or [])
            audit_feedback = str(state.get("audit_feedback") or "")
            is_revision = bool(audit_feedback)

            # Layer 1+2: reject when no evidence chunks available
            if not chunks:
                logger.info("generate_draft_node → insufficient evidence, writing refusal to draft")
                return {
                    "draft_answer": _INSUFFICIENT_EVIDENCE_MSG,
                    "claims": [],
                    "token_metrics": {},
                    "error_message": "",
                }

            # Assign short numeric IDs for LLM source reference
            chunk_id_map: dict[str, Any] = {}
            context_parts: list[str] = []
            for i, chunk in enumerate(chunks, 1):
                short_id = f"chunk-{i}"
                chunk_id_map[short_id] = chunk
                loc = (
                    f"{chunk.source_file}#page-{chunk.page_number}"
                    if getattr(chunk, "page_number", None)
                    else str(chunk.source_file)
                )
                context_parts.append(f"[ID: {short_id}] [{loc}]\n{chunk.text}")
            evidence_context = "\n\n".join(context_parts)

            # Base system prompt — no fallback to general knowledge
            system_text = (
                "You are a senior regulatory compliance analyst (Draftee). "
                "You MUST only use information explicitly present in the provided evidence chunks. "
                "If the evidence is insufficient to answer the question, state exactly: "
                "'当前文档库无法支撑此问题的回答' and do NOT fabricate any regulation articles, "
                "penalty clauses, case law, enforcement actions, or any fact not present in the evidence. "
                "Never cite regulation numbers, circular references, or legal provisions "
                "that do not appear verbatim in the provided chunks."
            )

            # Prompt body
            if is_revision:
                prompt = (
                    "Revise the compliance draft below based on the Auditor's feedback.\n\n"
                    f"## User Question\n{query}\n\n"
                    f"## Auditor Feedback\n{audit_feedback}\n\n"
                    f"## Evidence Context\n{evidence_context}\n\n"
                    "Return a JSON object with keys:\n"
                    '- answer_summary: Markdown regulatory analysis — address each point in the feedback.\n'
                    '- claims: Array of objects, each with:\n'
                    '    - statement: factual claim (one sentence)\n'
                    '    - source_ids: list of chunk IDs (e.g. ["chunk-1", "chunk-3"]) — REQUIRED, never empty\n'
                    '    - confidence: "high" | "medium" | "low"\n'
                    "IMPORTANT: Every claim.source_ids MUST reference valid chunk IDs from the evidence above. "
                    "Do NOT include claims that cannot be traced to a specific chunk."
                )
            else:
                prompt = (
                    "Answer the compliance question based ONLY on the evidence below.\n\n"
                    f"## User Question\n{query}\n\n"
                    f"## Evidence Context\n{evidence_context}\n\n"
                    "Return a JSON object with keys:\n"
                    '- answer_summary: Markdown regulatory analysis using ONLY evidence facts.\n'
                    '- claims: Array of objects, each with:\n'
                    '    - statement: factual claim (one sentence)\n'
                    '    - source_ids: list of chunk IDs (e.g. ["chunk-1", "chunk-3"]) — REQUIRED, never empty\n'
                    '    - confidence: "high" | "medium" | "low"\n'
                    "IMPORTANT: Every claim.source_ids MUST reference valid chunk IDs from the evidence above. "
                    "Do NOT include claims that cannot be traced to a specific chunk."
                )

            response = self._llm_request(self._draft_client, prompt, system_text)
            elapsed_ms = (time.perf_counter() - start) * 1000
            body = self._extract_text(response)
            token_delta = self._usage_tokens(response)
            logger.info(
                "generate_draft_node completed in %.1f ms [prompt=%d, completion=%d, total=%d]",
                elapsed_ms,
                token_delta.get("prompt_tokens", 0),
                token_delta.get("completion_tokens", 0),
                token_delta.get("total_tokens", 0),
            )

            # Accumulate token metrics
            prev_tokens = dict(state.get("token_metrics") or {})
            accumulated_tokens = {
                k: prev_tokens.get(k, 0) + token_delta.get(k, 0)
                for k in set(prev_tokens) | set(token_delta)
            }

            # Parse JSON response
            parsed = self._parse_json_response(body)
            if parsed is None:
                logger.warning("generate_draft_node → non-JSON response, treating as raw markdown")
                parsed = {"answer_summary": body, "claims": []}

            answer_summary = str(parsed.get("answer_summary", body))
            raw_claims: list[dict[str, Any]] = parsed.get("claims", [])
            if not isinstance(raw_claims, list):
                raw_claims = []

            # Remap short chunk IDs (e.g. "chunk-1") → real chunk IDs
            claim_real_ids: list[dict[str, Any]] = []
            for claim in raw_claims:
                short_ids = claim.get("source_ids", [])
                if isinstance(short_ids, list):
                    real_ids = [
                        str(getattr(chunk_id_map.get(sid), "chunk_id", sid))
                        for sid in short_ids
                    ]
                    claim["source_ids"] = real_ids
                claim_real_ids.append(claim)

            return {
                "draft_answer": answer_summary,
                "claims": claim_real_ids,
                "token_metrics": accumulated_tokens,
                "error_message": "",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("generate_draft_node failed: %s", exc)
            return {
                "draft_answer": "",
                "claims": [],
                "token_metrics": {},
                "error_message": f"Draft generation failed: {exc}",
            }

    # ------------------------------------------------------------------
    # Node 3 – auditor_review (Auditor role)
    # ------------------------------------------------------------------

    def auditor_review_node(self, state: AuditState) -> dict[str, Any]:
        try:
            start = time.perf_counter()
            query = str(state.get("user_query", ""))
            draft = str(state.get("draft_answer") or "")
            current_iter = int(state.get("current_iteration", 0))

            # Empty drafts are never approved implicitly.
            if not draft:
                logger.error("auditor_review_node → empty draft rejected")
                return {
                    "audit_feedback": "Draft is empty and cannot be audited.",
                    "audit_status": "error",
                    "current_iteration": current_iter + 1,
                    "error_message": "Auditor rejected an empty draft",
                }

            # Insufficient evidence draft → auto-approve
            if _INSUFFICIENT_EVIDENCE_MSG.strip() in draft:
                logger.info("auditor_review_node → insufficient evidence draft, auto-approved")
                return {
                    "audit_feedback": "",
                    "audit_status": "approved",
                    "current_iteration": current_iter + 1,
                    "error_message": "",
                }

            chunks = list(state.get("retrieved_chunks") or [])

            # Assign short IDs for auditor reference (must match generate_draft_node mapping)
            chunk_id_map: dict[str, Any] = {}
            for i, chunk in enumerate(chunks, 1):
                chunk_id_map[f"chunk-{i}"] = chunk

            # Build full evidence context for auditor (matching generate_draft_node format)
            evidence_parts: list[str] = []
            for short_id, chunk in chunk_id_map.items():
                loc = (
                    f"{chunk.source_file}#page-{chunk.page_number}"
                    if getattr(chunk, "page_number", None)
                    else str(chunk.source_file)
                )
                evidence_parts.append(f"[ID: {short_id}] [{loc}]\n{chunk.text}")
            evidence_context = "\n\n".join(evidence_parts)

            system_text = (
                "You are a senior compliance auditor. Your job is to review a Draftee's compliance answer "
                "for three critical issues:\n"
                "1. HALLUCINATION — did the Draftee invent facts, regulation numbers, penalties, or legal "
                "provisions NOT present in the evidence chunks?\n"
                "2. EVIDENCE GROUNDING — are all factual claims traceable to specific chunk IDs in the evidence?\n"
                "3. COMPLETENESS — does the answer adequately address the user's compliance question?\n\n"
                "Respond with STRICT JSON:\n"
                '- approved: true | false\n'
                '- feedback: string — if not approved, describe exactly what must be fixed. '
                'Be specific: quote the problematic claim, state why it is unsupported, '
                'and instruct how to revise it. If approved, set to "".\n'
                "IMPORTANT: Be strict. Approve only when there is zero hallucination, "
                "every claim is sourced, and the question is fully addressed."
            )

            prompt = (
                f"## User Question\n{query}\n\n"
                f"## Draftee Answer\n{draft}\n\n"
                f"## Evidence Context\n{evidence_context}"
            )

            response = self._llm_request(self._auditor_client, prompt, system_text)
            elapsed_ms = (time.perf_counter() - start) * 1000
            body = self._extract_text(response)
            token_delta = self._usage_tokens(response)
            previous_tokens = dict(state.get("token_metrics") or {})
            accumulated_tokens = {
                key: previous_tokens.get(key, 0) + token_delta.get(key, 0)
                for key in set(previous_tokens) | set(token_delta)
            }
            logger.info(
                "auditor_review_node completed in %.1f ms [approved determination]",
                elapsed_ms,
            )

            parsed = self._parse_json_response(body)
            if parsed is None:
                raise LLMError("Auditor response was not valid JSON")

            approved_value = parsed.get("approved")
            if not isinstance(approved_value, bool):
                raise LLMError("Auditor response is missing boolean 'approved'")
            approved = approved_value
            feedback = str(parsed.get("feedback", "")).strip()
            if not approved and not feedback:
                feedback = "Auditor rejected the draft without actionable feedback. Regenerate conservatively."

            if approved:
                logger.info("auditor_review_node → approved (iteration %d)", current_iter + 1)
            else:
                logger.info("auditor_review_node → feedback provided, iteration %d", current_iter + 1)

            return {
                "audit_feedback": "" if approved else feedback,
                "audit_status": "approved" if approved else "rejected",
                "current_iteration": current_iter + 1,
                "token_metrics": accumulated_tokens,
                "error_message": "",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("auditor_review_node failed: %s", exc)
            return {
                "audit_feedback": f"Auditor failure: {exc}",
                "audit_status": "error",
                "current_iteration": int(state.get("current_iteration", 0)) + 1,
                "error_message": f"Auditor review failed: {exc}",
            }

    # ------------------------------------------------------------------
    # Node 4 – prose grounding
    # ------------------------------------------------------------------

    def grounding_check_node(self, state: AuditState) -> dict[str, Any]:
        if not self._settings.inference.nli_enabled:
            return {"grounding_scores": {}, "error_message": ""}
        draft = str(state.get("draft_answer") or "")
        if _INSUFFICIENT_EVIDENCE_MSG.strip() in draft:
            return {"grounding_scores": {}, "error_message": ""}
        try:
            result = self._grounding_verifier.verify(
                draft,
                list(state.get("retrieved_chunks") or []),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("grounding_check_node failed: %s", exc)
            return {
                "audit_status": "error",
                "error_message": f"NLI grounding verification failed: {exc}",
            }
        if not result.passed:
            preview = "; ".join(result.unsupported_sentences[:3])
            return {
                "audit_status": "rejected",
                "audit_feedback": f"Unsupported prose detected by NLI: {preview}",
                "grounding_scores": result.scores,
                "error_message": "",
            }
        return {"grounding_scores": result.scores, "error_message": ""}

    def _grounding_router(self, state: AuditState) -> str:
        if state.get("error_message"):
            return "error_handler"
        return "finalize"

    # ------------------------------------------------------------------
    # Node 5 – finalize
    # ------------------------------------------------------------------

    def finalize_node(self, state: AuditState) -> dict[str, Any]:
        try:
            draft_answer = str(state.get("draft_answer") or "")
            raw_claims = list(state.get("claims") or [])
            chunks = list(state.get("retrieved_chunks") or [])

            # Layer 3+4: validate claims against actual retrieved chunk IDs
            valid_ids = _build_chunk_id_set(chunks)
            cite_map = _build_chunk_cite_map(chunks)
            validated_claims, cite_sources = _validate_claims(raw_claims, valid_ids, cite_map)

            audit_status = str(state.get("audit_status") or "pending")
            if audit_status == "rejected" and int(state.get("current_iteration", 0)) >= self._max_iterations:
                audit_status = "max_iterations"

            # Only approved content may cross the user-facing boundary.
            if audit_status == "approved" and draft_answer and _INSUFFICIENT_EVIDENCE_MSG.strip() not in draft_answer:
                final_output = draft_answer
            elif audit_status in {"rejected", "max_iterations"}:
                final_output = _UNAUDITED_REPORT_MSG
                validated_claims = []
                cite_sources = []
            elif not chunks:
                final_output = _INSUFFICIENT_EVIDENCE_MSG
            else:
                final_output = (
                    "# Compliance Q&A\n\n"
                    "No answer was generated. Please review the query and try again."
                )

            logger.info(
                "finalize_node → claims=%d (validated=%d), cite_sources=%d, output_len=%d",
                len(raw_claims),
                len(validated_claims),
                len(cite_sources),
                len(final_output),
            )
            return {
                "final_output": final_output,
                "claims": validated_claims,
                "cite_sources": cite_sources,
                "audit_status": audit_status,
                "error_message": "",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("finalize_node failed: %s", exc)
            return {
                "final_output": f"# Error Finalising\n\n{exc}",
                "error_message": f"Finalisation failed: {exc}",
            }

    # ------------------------------------------------------------------
    # Node 6 – error_handler
    # ------------------------------------------------------------------

    def error_handler_node(self, state: AuditState) -> dict[str, Any]:
        error_msg = str(state.get("error_message") or "Unknown error")
        logger.error("error_handler_node triggered: %s", error_msg)
        return {
            "final_output": (
                "# ⚠️ Compliance Q&A — Error\n\n"
                f"The following error occurred during processing:\n\n"
                f"**Error:** {error_msg}\n\n"
                "Please review the error and retry with corrected input."
            ),
            "claims": [],
            "cite_sources": [],
            "audit_status": "error",
            "error_message": error_msg,
        }

    # ------------------------------------------------------------------
    # Conditional routers
    # ------------------------------------------------------------------

    def _retrieve_router(self, state: AuditState) -> str:
        if state.get("error_message"):
            return "error_handler"
        return "generate_draft"

    def _auditor_router(self, state: AuditState) -> str:
        if state.get("error_message"):
            return "error_handler"
        if state.get("audit_status") == "approved":
            return "grounding_check"
        if state.get("current_iteration", 0) >= self._max_iterations:
            state["audit_status"] = "max_iterations"
            logger.info("auditor_router → max iterations (%d) reached, blocking draft", self._max_iterations)
            return "finalize"
        # Feedback present and iterations remaining → loop back to Draftee
        return "generate_draft"

    # ------------------------------------------------------------------
    # Build & invoke
    # ------------------------------------------------------------------

    def _ensure_compiled(self) -> Any:
        if self._compiled is not None:
            return self._compiled

        builder = StateGraph(AuditState)
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("generate_draft", self.generate_draft_node)
        builder.add_node("auditor_review", self.auditor_review_node)
        builder.add_node("grounding_check", self.grounding_check_node)
        builder.add_node("finalize", self.finalize_node)
        builder.add_node("error_handler", self.error_handler_node)

        # retrieve → generate_draft (or error_handler)
        builder.add_conditional_edges(
            "retrieve",
            self._retrieve_router,
            {
                "generate_draft": "generate_draft",
                "error_handler": "error_handler",
            },
        )
        builder.add_edge("generate_draft", "auditor_review")
        # auditor_review → generate_draft (loop) | finalize | error_handler
        builder.add_conditional_edges(
            "auditor_review",
            self._auditor_router,
            {
                "generate_draft": "generate_draft",
                "grounding_check": "grounding_check",
                "finalize": "finalize",
                "error_handler": "error_handler",
            },
        )
        builder.add_conditional_edges(
            "grounding_check",
            self._grounding_router,
            {"finalize": "finalize", "error_handler": "error_handler"},
        )
        builder.add_edge("finalize", END)
        builder.add_edge("error_handler", END)

        builder.set_entry_point("retrieve")
        self._compiled = builder.compile()
        logger.info(
            "ComplianceAgentGraph compiled (5-node Draft-Audit loop, max_iterations=%d)",
            self._max_iterations,
        )
        return self._compiled

    def invoke(self, state: AuditState) -> AuditResult:
        """Execute the compliance QA workflow synchronously.

        Parameters
        ----------
        state : AuditState
            Initial workflow state with at least `user_query`.

        Returns
        -------
        AuditResult
            Final result with answer, claims, citations, and token metrics.
        """
        self._ensure_compiled()

        initial: AuditState = {
            "user_query": "",
            "audit_intent": "compliance_qa",
            "retrieval_result": None,
            "retrieved_chunks": [],
            "current_iteration": 0,
            "draft_answer": "",
            "audit_feedback": "",
            "audit_status": "pending",
            "claims": [],
            "cite_sources": [],
            "error_message": "",
            "final_output": "",
            "token_metrics": {},
            "grounding_scores": {},
            **state,
        }

        try:
            final_state: AuditState = self._compiled.invoke(initial)
        except Exception as exc:  # noqa: BLE001
            logger.error("Graph invocation failed: %s", exc)
            final_state = {
                **initial,
                "error_message": f"Graph invocation failed: {exc}",
                "final_output": f"# Error\n\n{exc}",
            }

        answer = str(final_state.get("final_output") or "")
        claims = list(final_state.get("claims") or [])
        cite_sources = list(final_state.get("cite_sources") or [])
        token_metrics = dict(final_state.get("token_metrics") or {})
        error = str(final_state.get("error_message") or "") or None
        iterations = int(final_state.get("current_iteration", 0) or 0)
        audit_status = str(final_state.get("audit_status") or "pending")
        grounding_scores = dict(final_state.get("grounding_scores") or {})
        retrieved_chunks = list(final_state.get("retrieved_chunks") or []) or None

        return AuditResult(
            answer=answer,
            claims=claims,
            cite_sources=cite_sources,
            token_metrics=token_metrics,
            error=error,
            retrieved_chunks=retrieved_chunks,
            iterations=iterations,
            audit_status=audit_status,
            grounding_scores=grounding_scores,
        )


__all__ = [
    "AuditResult",
    "AuditState",
    "ComplianceAgentGraph",
]
