from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from anthropic import Anthropic

from config.settings import Settings, get_anthropic_client, get_settings
from src.indexing.milvus_ingest import BGEM3EmbeddingClient, MilvusHybridStore

logger = logging.getLogger(__name__)


class RetrievalDependencyError(RuntimeError):
    """Raised when retrieval dependencies are unavailable."""


class AuditIntent(str, Enum):
    COMPLIANCE_QA = "compliance_qa"
    TRANSACTION_AUDIT = "transaction_audit"


@dataclass
class AuditDecision:
    intent: AuditIntent
    rationale: str
    client_name: str | None = None
    violation_focus: str | None = None


@dataclass
class RetrievalRequest:
    user_query: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    filters: str | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    parent_id: str | None
    text: str
    source_file: str
    page_number: int | None
    chunk_type: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    audit_decision: AuditDecision
    expanded_queries: list[str]
    hyde_hypothesis: str
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    answer_context: str


class LLMProvider(Protocol):
    def classify_intent(self, query: str, history: list[dict[str, str]] | None = None) -> AuditDecision:
        ...

    def generate_multi_queries(self, query: str) -> list[str]:
        ...

    def generate_hyde_document(self, query: str) -> str:
        ...


class AnthropicLLMProvider:
    """Anthropic-backed provider for intent classification and query enrichment."""

    def __init__(self, settings: Settings | None = None, client: Anthropic | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or get_anthropic_client(self.settings)
        self._token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @property
    def token_usage(self) -> dict[str, int]:
        return dict(self._token_usage)

    def reset_token_usage(self) -> None:
        self._token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def classify_intent(self, query: str, history: list[dict[str, str]] | None = None) -> AuditDecision:
        prompt = self._build_intent_prompt(query=query, history=history or [])
        payload = self._invoke_claude(prompt)
        normalized_intent = str(payload.get("intent", AuditIntent.COMPLIANCE_QA.value)).strip().lower()
        intent = AuditIntent.TRANSACTION_AUDIT if normalized_intent == AuditIntent.TRANSACTION_AUDIT.value else AuditIntent.COMPLIANCE_QA
        return AuditDecision(
            intent=intent,
            rationale=str(payload.get("rationale", "No rationale provided.")).strip(),
            client_name=self._normalize_optional_text(payload.get("client_name")),
            violation_focus=self._normalize_optional_text(payload.get("violation_focus")),
        )

    def generate_multi_queries(self, query: str) -> list[str]:
        prompt = (
            "You expand compliance retrieval queries. Return strict JSON with key 'queries' as a list of 3 concise search rewrites. "
            "Preserve regulated-finance terminology and jurisdiction language.\n\n"
            f"Original query: {query}"
        )
        payload = self._invoke_claude(prompt)
        raw_queries = payload.get("queries", [])
        if not isinstance(raw_queries, list):
            return [query]
        cleaned = [str(item).strip() for item in raw_queries if str(item).strip()]
        return [query, *cleaned][:4]

    def generate_hyde_document(self, query: str) -> str:
        prompt = (
            "You generate a hypothetical compliance manual excerpt that would answer the user's question. "
            "Return strict JSON with key 'hypothesis'. Keep it factual, concise, and grounded in likely AML/KYC guidance.\n\n"
            f"Question: {query}"
        )
        payload = self._invoke_claude(prompt)
        hypothesis = str(payload.get("hypothesis", "")).strip()
        return hypothesis or query

    def _invoke_claude(self, prompt: str) -> dict[str, Any]:
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "You are a RegTech retrieval planner. Always return valid JSON and no extra text.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        request: dict[str, Any] = {
            "model": self.settings.llm.model,
            "max_tokens": self.settings.llm.max_tokens,
            "system": system_blocks,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.settings.llm.effort},
        }
        if self.settings.llm.enable_adaptive_thinking:
            request["thinking"] = {
                "type": "adaptive",
                "display": self.settings.llm.thinking_display,
            }

        max_attempts = 3
        delay_seconds = 1.0
        last_error: Exception | None = None
        raw_response: str = ""

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.messages.create(**request)
                self._accumulate_token_usage(response)
                raw_response = self._extract_text(response.content)
                try:
                    return json.loads(raw_response)
                except json.JSONDecodeError as exc:  # noqa: PERF203
                    last_error = exc
                    if attempt == max_attempts:
                        logger.warning(
                            "Claude response was not valid JSON after %s attempts: %s",
                            max_attempts,
                            raw_response,
                        )
                        return self._fallback_payload(prompt, raw_response)
                    continue
            except Exception as exc:  # noqa: BLE001
                if self._is_retryable_error(exc) and attempt < max_attempts:
                    last_error = exc
                    logger.warning(
                        "Claude request failed on attempt %s/%s: %s",
                        attempt,
                        max_attempts,
                        exc,
                    )
                    self._sleep_with_backoff(delay_seconds)
                    delay_seconds *= 2
                    continue
                raise RetrievalDependencyError(f"Claude request failed: {exc}") from exc

        raise RetrievalDependencyError("Claude response failed unexpectedly") from last_error

    def _build_intent_prompt(self, query: str, history: list[dict[str, str]]) -> str:
        serialized_history = json.dumps(history[-6:], ensure_ascii=False)
        return (
            "Classify the user's request for a compliance assistant. Return strict JSON with keys: "
            "intent ('compliance_qa' or 'transaction_audit'), rationale, client_name, violation_focus. "
            "Choose 'transaction_audit' only when the user wants transaction lookup, audit action, suspicious review, or report drafting.\n\n"
            f"History: {serialized_history}\n"
            f"Query: {query}"
        )

    def _extract_text(self, content: list[Any]) -> str:
        parts: list[str] = []
        for block in content:
            block_text = getattr(block, "text", None)
            if block_text:
                parts.append(str(block_text))
        return "\n".join(parts).strip()

    def _normalize_optional_text(self, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _accumulate_token_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = self._usage_value(usage, "input_tokens")
        completion_tokens = self._usage_value(usage, "output_tokens")
        total_tokens = self._usage_value(usage, "total_tokens")
        if prompt_tokens is None:
            prompt_tokens = 0
        if completion_tokens is None:
            completion_tokens = 0
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        self._token_usage["prompt_tokens"] += int(prompt_tokens)
        self._token_usage["completion_tokens"] += int(completion_tokens)
        self._token_usage["total_tokens"] += int(total_tokens)

    def _usage_value(self, usage: Any, key: str) -> int | None:
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        return int(value) if value is not None else None

    def _is_retryable_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        message = str(exc).lower()
        return "429" in message or "rate limit" in message or "too many requests" in message

    def _sleep_with_backoff(self, delay_seconds: float) -> None:
        time.sleep(delay_seconds)

    def _extract_prompt_value(self, prompt: str, label: str) -> str | None:
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith(label):
                value = stripped[len(label):].strip()
                return value or None
        return None

    def _fallback_payload(self, prompt: str, raw_response: str) -> dict[str, Any]:
        prompt_lower = prompt.lower()
        if "hypothesis" in prompt_lower:
            question = self._extract_prompt_value(prompt, "Question:")
            return {"hypothesis": question or raw_response.strip() or prompt}
        if "queries" in prompt_lower:
            original_query = self._extract_prompt_value(prompt, "Original query:")
            return {"queries": [original_query or raw_response.strip() or prompt]}
        if "intent" in prompt_lower:
            query = self._extract_prompt_value(prompt, "Query:")
            rationale = raw_response.strip()
            fallback_note = (
                f"Falling back to default compliance QA intent for: {query}"
                if query
                else "Falling back to default compliance QA intent."
            )
            if rationale:
                rationale = f"{fallback_note} Raw response: {rationale}"
            else:
                rationale = fallback_note
            return {
                "intent": AuditIntent.COMPLIANCE_QA.value,
                "rationale": rationale,
                "client_name": None,
                "violation_focus": None,
            }
        return {}


class CrossEncoderRerankerClient:
    """Cross-encoder reranker wrapper."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder as _CE

        settings = get_settings()
        self.model_name = model_name or settings.inference.reranker_model_name
        self._CE = _CE
        self.model = _CE(self.model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int, score_threshold: float = 0.0) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        rescored = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                text=chunk.text,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                chunk_type=chunk.chunk_type,
                score=float(score),
                metadata=chunk.metadata,
            )
            for chunk, score in zip(chunks, scores)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        if score_threshold > 0.0:
            rescored = [c for c in rescored if c.score >= score_threshold]
        return rescored[:top_k]


class ComplianceRetrievalPipeline:
    """Hybrid retrieval pipeline with query expansion, HyDE, RRF, and reranking."""

    def __init__(
        self,
        store: MilvusHybridStore | None = None,
        embedding_client: BGEM3EmbeddingClient | None = None,
        reranker: CrossEncoderRerankerClient | None = None,
        llm_provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or MilvusHybridStore(self.settings)
        self.embedding_client = embedding_client or BGEM3EmbeddingClient()
        self.reranker = reranker  # None = skip reranking, use RRF fusion only
        self.llm_provider = llm_provider or AnthropicLLMProvider(self.settings)

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        audit_decision = await asyncio.to_thread(
            self.llm_provider.classify_intent,
            request.user_query,
            request.conversation_history,
        )
        expanded_queries = await asyncio.to_thread(self.llm_provider.generate_multi_queries, request.user_query)
        hyde_hypothesis = await asyncio.to_thread(self.llm_provider.generate_hyde_document, request.user_query)

        search_queries = self._deduplicate_queries([request.user_query, *expanded_queries, hyde_hypothesis])
        search_results = await self._run_parallel_searches(search_queries, request.filters)
        fused_chunks = self._reciprocal_rank_fusion(search_results)
        if self.reranker is not None:
            reranked_chunks = await asyncio.to_thread(
                self.reranker.rerank,
                request.user_query,
                fused_chunks,
                self.settings.retrieval.rerank_top_k,
                self.settings.retrieval.rerank_score_threshold,
            )
        else:
            reranked_chunks = fused_chunks[:self.settings.retrieval.rerank_top_k]
        answer_context = self._build_answer_context(reranked_chunks)
        return RetrievalResult(
            audit_decision=audit_decision,
            expanded_queries=expanded_queries,
            hyde_hypothesis=hyde_hypothesis,
            retrieved_chunks=fused_chunks,
            reranked_chunks=reranked_chunks,
            answer_context=answer_context,
        )

    async def _run_parallel_searches(self, queries: list[str], filters: str | None) -> list[list[RetrievedChunk]]:
        semaphore = asyncio.Semaphore(self.settings.retrieval.async_workers)

        async def _search(query: str) -> list[RetrievedChunk]:
            async with semaphore:
                return await asyncio.to_thread(self._search_single_query, query, filters)

        tasks = [_search(query) for query in queries]
        return await asyncio.gather(*tasks)

    def _search_single_query(self, query: str, filters: str | None) -> list[RetrievedChunk]:
        embedding = self.embedding_client.encode(query)
        dense_hits, sparse_hits = self.store.hybrid_search(
            dense_vector=embedding.dense_vector,
            sparse_vector=embedding.sparse_vector,
            top_k=max(self.settings.retrieval.dense_top_k, self.settings.retrieval.sparse_top_k),
            output_fields=[
                "chunk_id",
                "parent_id",
                "chunk_type",
                "source_file",
                "page_number",
                "text",
                "metadata_json",
            ],
            filters=filters,
        )
        dense_chunks = self._to_chunks(dense_hits[: self.settings.retrieval.dense_top_k])
        sparse_chunks = self._to_chunks(sparse_hits[: self.settings.retrieval.sparse_top_k])
        return [*dense_chunks, *sparse_chunks]

    def _to_chunks(self, hits: list[dict[str, Any]]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=str(hit["chunk_id"]),
                parent_id=hit.get("parent_id"),
                text=str(hit["text"]),
                source_file=str(hit["source_file"]),
                page_number=hit.get("page_number"),
                chunk_type=str(hit["chunk_type"]),
                score=float(hit["score"]),
                metadata=dict(hit.get("metadata", {})),
            )
            for hit in hits
        ]

    def _reciprocal_rank_fusion(self, result_sets: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        scored: dict[str, float] = {}
        canonical: dict[str, RetrievedChunk] = {}
        for result_set in result_sets:
            for rank, chunk in enumerate(result_set, start=1):
                scored[chunk.chunk_id] = scored.get(chunk.chunk_id, 0.0) + 1.0 / (self.settings.retrieval.rrf_k + rank)
                canonical.setdefault(chunk.chunk_id, chunk)

        fused = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                text=chunk.text,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                chunk_type=chunk.chunk_type,
                score=scored[chunk.chunk_id],
                metadata=chunk.metadata,
            )
            for chunk in canonical.values()
        ]
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[: self.settings.retrieval.rrf_top_k]

    def _build_answer_context(self, chunks: list[RetrievedChunk]) -> str:
        sections: list[str] = []
        for chunk in chunks:
            location = f"{chunk.source_file}#page-{chunk.page_number}" if chunk.page_number else chunk.source_file
            sections.append(f"[{location}]\n{chunk.text}")
        return "\n\n".join(sections)

    def _deduplicate_queries(self, queries: list[str]) -> list[str]:
        seen: set[str] = set()
        deduplicated: list[str] = []
        for query in queries:
            normalized = query.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(normalized)
        return deduplicated


__all__ = [
    "AnthropicLLMProvider",
    "AuditDecision",
    "AuditIntent",
    "ComplianceRetrievalPipeline",
    "CrossEncoderRerankerClient",
    "RetrievedChunk",
    "RetrievalDependencyError",
    "RetrievalRequest",
    "RetrievalResult",
]
