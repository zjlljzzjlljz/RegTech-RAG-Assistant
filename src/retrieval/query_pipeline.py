from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import httpx

from config.settings import Settings, get_settings
from src.indexing.milvus_ingest import BGEEmbeddingResult, BGEM3EmbeddingClient, MilvusHybridStore
from src.inference import LLMClient, create_llm_client
from src.inference.llm_client import LLMError, parse_json_object

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
    initial_retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    initial_reranked_chunks: list[RetrievedChunk] = field(default_factory=list)


class LLMProvider(Protocol):
    def classify_intent(self, query: str, history: list[dict[str, str]] | None = None) -> AuditDecision:
        ...

    def generate_multi_queries(self, query: str) -> list[str]:
        ...

    def generate_hyde_document(self, query: str) -> str:
        ...


class ConfigurableLLMProvider:
    """Provider-neutral query planner backed by the configured planner model."""

    def __init__(self, settings: Settings | None = None, client: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or create_llm_client(self.settings.llm_roles.planner, self.settings)
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
        payload = self._invoke_json(prompt)
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
            "Preserve the jurisdiction and use exact regulated-finance terminology, common acronyms, and likely wording from official guidance. "
            "If the original query contains Chinese, produce at least 2 English keyword-rich rewrites suitable for searching an English regulatory corpus. "
            "Do not return three near-duplicate paraphrases.\n\n"
            f"Original query: {query}"
        )
        payload = self._invoke_json(prompt)
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
        payload = self._invoke_json(prompt)
        hypothesis = str(payload.get("hypothesis", "")).strip()
        return hypothesis or query

    def _invoke_json(self, prompt: str) -> dict[str, Any]:
        max_attempts = 3
        delay_seconds = 1.0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.complete(
                    system="You are a RegTech retrieval planner. Return valid JSON and no extra text.",
                    prompt=prompt,
                    json_mode=True,
                )
                self._accumulate_token_usage(response)
                try:
                    return parse_json_object(response.text)
                except LLMError as exc:  # noqa: PERF203
                    last_error = exc
                    if attempt == max_attempts:
                        raise RetrievalDependencyError("Planner returned invalid JSON") from exc
                    continue
            except Exception as exc:  # noqa: BLE001
                if self._is_retryable_error(exc) and attempt < max_attempts:
                    last_error = exc
                    logger.warning(
                        "Planner request failed on attempt %s/%s: %s",
                        attempt,
                        max_attempts,
                        exc,
                    )
                    self._sleep_with_backoff(delay_seconds)
                    delay_seconds *= 2
                    continue
                if isinstance(exc, RetrievalDependencyError):
                    raise
                raise RetrievalDependencyError(f"Planner request failed: {exc}") from exc

        raise RetrievalDependencyError("Planner response failed unexpectedly") from last_error

    def _build_intent_prompt(self, query: str, history: list[dict[str, str]]) -> str:
        serialized_history = json.dumps(history[-6:], ensure_ascii=False)
        return (
            "Classify the user's request for a compliance assistant. Return strict JSON with keys: "
            "intent ('compliance_qa' or 'transaction_audit'), rationale, client_name, violation_focus. "
            "Choose 'transaction_audit' only when the user wants transaction lookup, audit action, suspicious review, or report drafting.\n\n"
            f"History: {serialized_history}\n"
            f"Query: {query}"
        )

    def _normalize_optional_text(self, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _accumulate_token_usage(self, response: Any) -> None:
        prompt_tokens = int(getattr(response, "prompt_tokens", 0))
        completion_tokens = int(getattr(response, "completion_tokens", 0))
        total_tokens = int(getattr(response, "total_tokens", prompt_tokens + completion_tokens))
        self._token_usage["prompt_tokens"] += int(prompt_tokens)
        self._token_usage["completion_tokens"] += int(completion_tokens)
        self._token_usage["total_tokens"] += int(total_tokens)

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

AnthropicLLMProvider = ConfigurableLLMProvider


class CrossEncoderRerankerClient:
    """Cross-encoder reranker wrapper."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.inference.reranker_model_name
        self.service_url = settings.inference.reranker_service_url
        self.timeout = settings.inference.request_timeout_seconds
        self.model: Any = None
        if not self.service_url:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int, score_threshold: float = 0.0) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if self.service_url:
            response = httpx.post(
                f"{self.service_url.rstrip('/')}/rerank",
                json={"query": query, "documents": [chunk.text for chunk in chunks]},
                timeout=self.timeout,
            )
            response.raise_for_status()
            scores = response.json()["scores"]
        else:
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
        self.llm_provider = llm_provider or ConfigurableLLMProvider(self.settings)
        self._planner_cache: dict[str, tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        intent_task = asyncio.to_thread(
            self._cached_planner_call,
            "intent",
            request.user_query,
            request.conversation_history,
        )
        if self._should_expand_query(request.user_query):
            queries_task = asyncio.to_thread(
                self._cached_planner_call,
                "queries",
                request.user_query,
                request.conversation_history,
            )
            audit_decision, expanded_queries = await asyncio.gather(intent_task, queries_task)
        else:
            audit_decision = await intent_task
            expanded_queries = [request.user_query]

        search_queries = self._deduplicate_queries([request.user_query, *expanded_queries])
        search_results = await self._run_parallel_searches(search_queries, request.filters)
        search_weights = []
        for result_index in range(len(search_results)):
            query_weight = self.settings.retrieval.original_query_rrf_weight if result_index < 2 else 1.0
            arm_weight = (
                self.settings.retrieval.dense_rrf_weight
                if result_index % 2 == 0
                else self.settings.retrieval.sparse_rrf_weight
            )
            search_weights.append(query_weight * arm_weight)
        fused_chunks = self._reciprocal_rank_fusion(search_results, search_weights)
        reranked_chunks = await self._rerank(request.user_query, fused_chunks)
        initial_retrieved_chunks = list(fused_chunks)
        initial_reranked_chunks = list(reranked_chunks)

        hyde_hypothesis = ""
        if self._should_trigger_hyde(reranked_chunks):
            hyde_hypothesis = await asyncio.to_thread(
                self._cached_planner_call,
                "hyde",
                request.user_query,
                request.conversation_history,
            )
            if hyde_hypothesis:
                hyde_results = await self._run_parallel_searches([hyde_hypothesis], request.filters)
                search_results.extend(hyde_results)
                search_weights.extend(
                    self.settings.retrieval.dense_rrf_weight
                    if result_index % 2 == 0
                    else self.settings.retrieval.sparse_rrf_weight
                    for result_index in range(len(hyde_results))
                )
                fused_chunks = self._reciprocal_rank_fusion(search_results, search_weights)
                reranked_chunks = await self._rerank(request.user_query, fused_chunks)

        if self.settings.retrieval.parent_backfill:
            reranked_chunks = await asyncio.to_thread(self._backfill_parent_context, reranked_chunks)

        answer_context = self._build_answer_context(reranked_chunks)
        return RetrievalResult(
            audit_decision=audit_decision,
            expanded_queries=expanded_queries,
            hyde_hypothesis=hyde_hypothesis,
            retrieved_chunks=fused_chunks,
            reranked_chunks=reranked_chunks,
            answer_context=answer_context,
            initial_retrieved_chunks=initial_retrieved_chunks,
            initial_reranked_chunks=initial_reranked_chunks,
        )

    async def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self.reranker is not None:
            return await asyncio.to_thread(
                self.reranker.rerank,
                query,
                chunks,
                self.settings.retrieval.rerank_top_k,
                self.settings.retrieval.rerank_score_threshold,
            )
        return chunks[:self.settings.retrieval.rerank_top_k]

    def _should_trigger_hyde(self, chunks: list[RetrievedChunk]) -> bool:
        if self.reranker is None:
            return False
        if not chunks:
            return True
        top_score = chunks[0].score
        margin = top_score - chunks[1].score if len(chunks) > 1 else top_score
        return (
            top_score < self.settings.retrieval.hyde_score_threshold
            or margin < self.settings.retrieval.hyde_margin_threshold
        )

    def _cached_planner_call(
        self,
        operation: str,
        query: str,
        history: list[dict[str, str]],
    ) -> Any:
        history_key = json.dumps(history[-6:], ensure_ascii=False, sort_keys=True) if operation == "intent" else ""
        key = f"{operation}:{query.strip().lower()}:{history_key}"
        now = time.monotonic()
        with self._cache_lock:
            cached = self._planner_cache.get(key)
            if cached and cached[0] > now:
                return cached[1]

        if operation == "intent":
            value = self.llm_provider.classify_intent(query, history)
        elif operation == "queries":
            value = self.llm_provider.generate_multi_queries(query)
        elif operation == "hyde":
            value = self.llm_provider.generate_hyde_document(query)
        else:
            raise ValueError(f"Unknown planner operation: {operation}")

        expires_at = now + self.settings.retrieval.planner_cache_ttl_seconds
        with self._cache_lock:
            self._planner_cache[key] = (expires_at, value)
        return value

    def _backfill_parent_context(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        parent_ids = list(dict.fromkeys(chunk.parent_id for chunk in chunks if chunk.parent_id))
        if not parent_ids:
            return chunks
        parents = self.store.get_chunks_by_ids(parent_ids)
        parent_map = {str(item["chunk_id"]): item for item in parents}
        output: list[RetrievedChunk] = []
        seen_parent_ids: set[str] = set()
        for chunk in chunks:
            parent = parent_map.get(str(chunk.parent_id))
            if parent is None:
                output.append(chunk)
                continue
            parent_key = str(chunk.parent_id)
            if parent_key in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_key)
            metadata = dict(parent.get("metadata", {}))
            metadata["matched_child_id"] = chunk.chunk_id
            output.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    parent_id=chunk.parent_id,
                    text=str(parent["text"]),
                    source_file=str(parent["source_file"]),
                    page_number=parent.get("page_number"),
                    chunk_type="parent_context",
                    score=chunk.score,
                    metadata=metadata,
                )
            )
        return output

    async def _run_parallel_searches(self, queries: list[str], filters: str | None) -> list[list[RetrievedChunk]]:
        semaphore = asyncio.Semaphore(self.settings.retrieval.async_workers)

        if hasattr(self.embedding_client, "encode_many"):
            embeddings = await asyncio.to_thread(
                self.embedding_client.encode_many,
                queries,
                "Represent this sentence for searching relevant passages: ",
            )

            async def _search_embedding(embedding: BGEEmbeddingResult) -> list[list[RetrievedChunk]]:
                async with semaphore:
                    return await asyncio.to_thread(self._search_with_embedding, embedding, filters)

            ranked_pairs = await asyncio.gather(*(_search_embedding(embedding) for embedding in embeddings))
            return [ranked_list for pair in ranked_pairs for ranked_list in pair]

        async def _search(query: str) -> list[list[RetrievedChunk]]:
            async with semaphore:
                return await asyncio.to_thread(self._search_single_query, query, filters)

        tasks = [_search(query) for query in queries]
        ranked_pairs = await asyncio.gather(*tasks)
        return [ranked_list for pair in ranked_pairs for ranked_list in pair]

    def _search_single_query(self, query: str, filters: str | None) -> list[list[RetrievedChunk]]:
        embedding = self.embedding_client.encode(
            query,
            prompt="Represent this sentence for searching relevant passages: ",
        )
        return self._search_with_embedding(embedding, filters)

    def _search_with_embedding(
        self,
        embedding: BGEEmbeddingResult,
        filters: str | None,
    ) -> list[list[RetrievedChunk]]:
        child_filter = 'chunk_type == "child"'
        effective_filters = f"({filters}) and {child_filter}" if filters else child_filter
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
            filters=effective_filters,
        )
        dense_chunks = self._to_chunks(dense_hits[: self.settings.retrieval.dense_top_k])
        sparse_chunks = self._to_chunks(sparse_hits[: self.settings.retrieval.sparse_top_k])
        return [dense_chunks, sparse_chunks]

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

    def _reciprocal_rank_fusion(
        self,
        result_sets: list[list[RetrievedChunk]],
        weights: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        scored: dict[str, float] = {}
        canonical: dict[str, RetrievedChunk] = {}
        effective_weights = weights or [1.0] * len(result_sets)
        if len(effective_weights) != len(result_sets):
            raise ValueError("RRF weights must match the number of result sets")
        for result_set, weight in zip(result_sets, effective_weights):
            for rank, chunk in enumerate(result_set, start=1):
                scored[chunk.chunk_id] = scored.get(chunk.chunk_id, 0.0) + weight / (self.settings.retrieval.rrf_k + rank)
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

    def _should_expand_query(self, query: str) -> bool:
        if not re.search(r"[\u3400-\u9fff]", query):
            return False
        latin_terms = re.findall(r"[A-Za-z][A-Za-z-]{2,}", query)
        return len(latin_terms) < 4


__all__ = [
    "AnthropicLLMProvider",
    "ConfigurableLLMProvider",
    "AuditDecision",
    "AuditIntent",
    "ComplianceRetrievalPipeline",
    "CrossEncoderRerankerClient",
    "RetrievedChunk",
    "RetrievalDependencyError",
    "RetrievalRequest",
    "RetrievalResult",
]
