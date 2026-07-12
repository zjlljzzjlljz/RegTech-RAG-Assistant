#!/usr/bin/env python
"""Generation evaluation script for compliance RAG pipeline.

Usage:
    python -m src.evaluation.eval_generation

Runs the full compliance agent graph over annotated test queries and evaluates
answer quality with RAGAs metrics for faithfulness, answer relevancy, and
context recall.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import configure_logging, get_settings
from src.agent import ComplianceAgentGraph
from src.indexing import BGEM3EmbeddingClient, MilvusHybridStore
from src.inference import create_llm_client
from src.retrieval import ComplianceRetrievalPipeline, CrossEncoderRerankerClient

logger = logging.getLogger(__name__)
_DEBUG_RAGAS = os.getenv("RAGAS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_LIMIT = int(os.getenv("RAGAS_DEBUG_LIMIT", "0") or "0")

try:
    from datasets import Dataset
except Exception as exc:  # noqa: BLE001
    Dataset = None  # type: ignore[assignment]
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None

try:
    from langchain_core.outputs import Generation, LLMResult
    from ragas import evaluate
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import BaseRagasLLM
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
except Exception as exc:  # noqa: BLE001
    Generation = None  # type: ignore[assignment]
    LLMResult = None  # type: ignore[assignment]
    evaluate = None  # type: ignore[assignment]
    BaseRagasEmbeddings = None  # type: ignore[assignment]
    BaseRagasLLM = None  # type: ignore[assignment]
    Faithfulness = None  # type: ignore[assignment]
    LLMContextRecall = None  # type: ignore[assignment]
    ResponseRelevancy = None  # type: ignore[assignment]
    _RAGAS_IMPORT_ERROR = exc
else:
    _RAGAS_IMPORT_ERROR = None


def _prompt_to_text(prompt: Any) -> str:
    if hasattr(prompt, "to_string"):
        return str(prompt.to_string())
    return str(prompt)


def _response_text(content: Any) -> str:
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


if BaseRagasLLM is not None and Generation is not None and LLMResult is not None:

    class AnthropicRagasLLM(BaseRagasLLM):
        """RAGAs LLM judge backed by the project's Anthropic client."""

        def __init__(self, client: Any, model_name: str, max_tokens: int, effort: str) -> None:
            self._client = client
            self.model_name = model_name
            self.max_tokens = max_tokens
            self.effort = effort
            self.run_config: Any = None

        def set_run_config(self, run_config: Any) -> None:
            self.run_config = run_config

        def get_temperature(self, n: int = 1) -> float:
            return 0.0

        def is_finished(self, response: LLMResult) -> bool:
            return True

        def _request_text(
            self,
            prompt: Any,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
        ) -> str:
            response = self._client.complete(
                system=(
                    "You are a strict evaluator for a regulated-finance retrieval benchmark. "
                    "Follow the prompt exactly and return only the requested output."
                ),
                prompt=_prompt_to_text(prompt),
            )
            text = response.text
            if _DEBUG_RAGAS:
                logger.info(
                    "RAGAS_DEBUG _request_text -> chars=%d empty=%s stop=%s",
                    len(text),
                    not bool(text.strip()),
                    bool(stop),
                )
                if text:
                    logger.info("RAGAS_DEBUG _request_text preview=%r", text[:240])
            return text

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Any = None,
            **kwargs: Any,
        ) -> LLMResult:
            text = self._request_text(prompt, temperature=temperature, stop=stop)
            generations = [Generation(text=text) for _ in range(max(1, n))]
            result = LLMResult(generations=[generations], llm_output={"model": self.model_name})
            if _DEBUG_RAGAS:
                outer = len(result.generations)
                inner = len(result.generations[0]) if result.generations else 0
                first = result.generations[0][0].text if result.generations and result.generations[0] else ""
                logger.info(
                    "RAGAS_DEBUG generate_text -> requested_n=%d outer=%d inner=%d first_chars=%d kwargs=%s",
                    n,
                    outer,
                    inner,
                    len(first),
                    sorted(kwargs.keys()),
                )
            return result

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Any = None,
            **kwargs: Any,
        ) -> LLMResult:
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n,
                temperature,
                stop,
                callbacks,
                **kwargs,
            )

else:

    class AnthropicRagasLLM:
        """Fallback LLM adapter used when RAGAs is not installed."""

        def __init__(self, client: Any, model_name: str, max_tokens: int, effort: str) -> None:
            self._client = client
            self.model_name = model_name
            self.max_tokens = max_tokens
            self.effort = effort
            self.run_config: Any = None

        def set_run_config(self, run_config: Any) -> None:
            self.run_config = run_config

        def get_temperature(self, n: int = 1) -> float:
            return 0.0

        def _request_text(
            self,
            prompt: Any,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
        ) -> str:
            response = self._client.complete(
                system=(
                    "You are a strict evaluator for a regulated-finance retrieval benchmark. "
                    "Follow the prompt exactly and return only the requested output."
                ),
                prompt=_prompt_to_text(prompt),
            )
            return response.text

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Any = None,
            **kwargs: Any,
        ) -> Any:
            return {
                "generations": [[{"text": self._request_text(prompt, temperature=temperature, stop=stop)}]],
                "llm_output": {"model": self.model_name},
            }

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Any = None,
            **kwargs: Any,
        ) -> Any:
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n,
                temperature,
                stop,
                callbacks,
                **kwargs,
            )


if BaseRagasEmbeddings is not None:

    class BGEM3RagasEmbeddings(BaseRagasEmbeddings):
        """RAGAs embeddings adapter backed by BGEM3."""

        def __init__(self, embedding_client: BGEM3EmbeddingClient) -> None:
            self._embedding_client = embedding_client
            self.run_config: Any = None

        def set_run_config(self, run_config: Any) -> None:
            self.run_config = run_config

        def _embed_many(self, texts: list[str], prompt: str | None = None) -> list[list[float]]:
            vectors: list[list[float]] = []
            for text in texts:
                result = self._embedding_client.encode(text, prompt=prompt)
                vectors.append(list(result.dense_vector))
            return vectors

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return self._embed_many(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._embed_many(
                [text],
                prompt="Represent this sentence for searching relevant passages: ",
            )[0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embed_many(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

else:

    class BGEM3RagasEmbeddings:
        """Fallback embeddings adapter used when RAGAs is not installed."""

        def __init__(self, embedding_client: BGEM3EmbeddingClient) -> None:
            self._embedding_client = embedding_client
            self.run_config: Any = None

        def set_run_config(self, run_config: Any) -> None:
            self.run_config = run_config

        def _embed_many(self, texts: list[str], prompt: str | None = None) -> list[list[float]]:
            vectors: list[list[float]] = []
            for text in texts:
                result = self._embedding_client.encode(text, prompt=prompt)
                vectors.append(list(result.dense_vector))
            return vectors

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return self._embed_many(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._embed_many(
                [text],
                prompt="Represent this sentence for searching relevant passages: ",
            )[0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embed_many(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)


def _run_coroutine(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "asyncio.run() cannot be called from a running event loop" not in message and "event loop" not in message:
            raise
        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def load_test_queries(filepath: Path) -> list[dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(filepath, encoding="utf-8") as fh:
        queries = json.load(fh)
    logger.info("Loaded %d test queries from %s", len(queries), filepath)
    return queries


def build_agent_stack(
    settings: Any,
) -> tuple[ComplianceAgentGraph, ComplianceRetrievalPipeline, BGEM3EmbeddingClient]:
    """Build the compliance graph and retrieval pipeline using the app pattern."""
    store = MilvusHybridStore(settings)
    embedding_client = BGEM3EmbeddingClient(settings.inference.embedding_model_name)

    reranker = None
    if settings.inference.reranker_enabled:
        try:
            reranker = CrossEncoderRerankerClient(settings.inference.reranker_model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker unavailable, using RRF retrieval mode: %s", exc)

    pipeline = ComplianceRetrievalPipeline(
        store=store,
        embedding_client=embedding_client,
        reranker=reranker,
        settings=settings,
    )
    agent_graph = ComplianceAgentGraph(retrieval_pipeline=pipeline, settings=settings)
    return agent_graph, pipeline, embedding_client


def _extract_contexts_from_chunks(chunks: list[Any] | None) -> list[str]:
    contexts: list[str] = []
    for chunk in list(chunks or []):
        text = str(getattr(chunk, "text", "")).strip()
        if text:
            contexts.append(text)
    return contexts


def run_generation_eval(
    test_queries: list[dict[str, Any]],
    agent_graph: ComplianceAgentGraph,
) -> list[dict[str, Any]]:
    """Run the graph against annotated queries and collect evaluation records."""
    records: list[dict[str, Any]] = []
    annotated_queries = [q for q in test_queries if str(q.get("ground_truth_answer", "")).strip()]
    if _DEBUG_LIMIT > 0:
        annotated_queries = annotated_queries[:_DEBUG_LIMIT]

    for index, item in enumerate(annotated_queries, start=1):
        query = str(item.get("query", "")).strip()
        if not query:
            continue

        sample_start = time.perf_counter()
        logger.info("[%d/%d] Evaluating generation: %s", index, len(annotated_queries), query[:80])

        retrieval_ms = 0.0
        try:
            retrieval_start = time.perf_counter()
            result = agent_graph.invoke({"user_query": query})
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent evaluation failed for query '%s': %s", query[:60], exc)
            continue

        contexts = _extract_contexts_from_chunks(getattr(result, "retrieved_chunks", None))
        graph_ms = retrieval_ms
        answer = str(getattr(result, "answer", "") or "")

        record = {
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": str(item.get("ground_truth_answer", "") or "").strip(),
            "audit_status": str(getattr(result, "audit_status", "pending")),
            "claims": list(getattr(result, "claims", []) or []),
            "grounding_scores": dict(getattr(result, "grounding_scores", {}) or {}),
        }
        if _DEBUG_RAGAS:
            logger.info(
                "RAGAS_DEBUG record -> query_chars=%d answer_chars=%d contexts=%d ground_truth_chars=%d",
                len(record["question"]),
                len(record["answer"]),
                len(record["contexts"]),
                len(record["ground_truth"]),
            )

        elapsed_ms = (time.perf_counter() - sample_start) * 1000
        logger.info(
            "[%d/%d] sample done in %.1f ms (graph=%.1f ms, contexts=%d)",
            index,
            len(annotated_queries),
            elapsed_ms,
            graph_ms,
            len(contexts),
        )
        records.append(record)

    return records


def _format_bar(value: float, width: int = 24) -> str:
    if not math.isfinite(value):
        return "?" * width
    filled = max(0, min(width, int(round(value * width))))
    return "#" * filled + "." * (width - filled)


def print_report(scores: dict[str, float], evaluated_count: int) -> None:
    """Print evaluation report to stdout."""
    print("\n" + "=" * 72)
    print("  Compliance Generation Evaluation Report")
    print("=" * 72)
    print(f"  Evaluated queries: {evaluated_count}")

    for label, key in (
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Recall", "context_recall"),
    ):
        score = float(scores.get(key, 0.0))
        score_text = f"{score:.4f}" if math.isfinite(score) else "nan"
        print(f"\n  {label:17} {score_text:>6}  [{_format_bar(score)}]")

    print("\n" + "=" * 72)


def _extract_metric_score(result: Any, metric_name: str) -> float:
    scores = getattr(result, "scores", None)
    if isinstance(scores, list):
        values: list[float] = []
        for row in scores:
            if not isinstance(row, dict):
                continue
            value = row.get(metric_name)
            if value is None:
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                values.append(numeric)
        if values:
            return sum(values) / len(values)
        return float("nan")

    if isinstance(scores, dict):
        value = scores.get(metric_name)
        if value is not None:
            return float(value)

    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            if metric_name in df.columns:
                return float(df[metric_name].mean())
        except Exception:  # noqa: BLE001
            pass

    if isinstance(result, dict):
        value = result.get(metric_name)
        if value is not None:
            return float(value)

    return 0.0


def _evaluate_ragas(
    records: list[dict[str, Any]],
    settings: Any,
    embedding_client: BGEM3EmbeddingClient,
) -> dict[str, float]:
    """Run RAGAs evaluation over the collected records."""
    if not records:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}

    if Dataset is None:
        raise RuntimeError(f"datasets import failed: {_DATASETS_IMPORT_ERROR}")
    if evaluate is None or BaseRagasLLM is None or BaseRagasEmbeddings is None:
        raise RuntimeError(f"ragas import failed: {_RAGAS_IMPORT_ERROR}")
    if Faithfulness is None or ResponseRelevancy is None or LLMContextRecall is None:
        raise RuntimeError(f"ragas metrics import failed: {_RAGAS_IMPORT_ERROR}")

    hf_dataset = Dataset.from_list(records)

    llm_client = create_llm_client(settings.llm_roles.judge, settings)
    llm = AnthropicRagasLLM(
        llm_client,
        settings.llm_roles.judge.model,
        settings.llm_roles.judge.max_tokens,
        "high",
    )
    embeddings = BGEM3RagasEmbeddings(embedding_client)

    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextRecall()]
    result = evaluate(
        hf_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    if _DEBUG_RAGAS:
        raw_scores = getattr(result, "scores", None)
        logger.info("RAGAS_DEBUG evaluate result type=%s", type(result).__name__)
        logger.info("RAGAS_DEBUG evaluate scores type=%s", type(raw_scores).__name__)
        logger.info("RAGAS_DEBUG evaluate scores raw=%r", raw_scores)
        for metric_name in ("faithfulness", "answer_relevancy", "context_recall"):
            logger.info(
                "RAGAS_DEBUG aggregated %s=%s",
                metric_name,
                _extract_metric_score(result, metric_name),
            )

    return {
        "faithfulness": _extract_metric_score(result, "faithfulness"),
        "answer_relevancy": _extract_metric_score(result, "answer_relevancy"),
        "context_recall": _extract_metric_score(result, "context_recall"),
    }


def _safety_scores(records: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    total_claims = 0
    cited_claims = 0
    nli_scores: list[float] = []
    unsafe_passes = 0
    for record in records:
        for claim in record.get("claims", []):
            total_claims += 1
            if claim.get("source_ids"):
                cited_claims += 1
        scores = [float(value) for value in record.get("grounding_scores", {}).values()]
        nli_scores.extend(scores)
        if record.get("audit_status") == "approved" and any(score < threshold for score in scores):
            unsafe_passes += 1
    return {
        "citation_precision": cited_claims / total_claims if total_claims else 1.0,
        "unsupported_sentence_rate": (
            sum(score < threshold for score in nli_scores) / len(nli_scores) if nli_scores else 0.0
        ),
        "unsafe_auditor_pass_rate": unsafe_passes / len(records) if records else 0.0,
    }


def write_results(output_dir: Path, scores: dict[str, float], evaluated_count: int) -> Path | None:
    """Persist evaluation summary to JSON."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluated_queries": evaluated_count,
        "scores": scores,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"ragas_eval_{timestamp}.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return output_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write JSON results to %s: %s", output_path, exc)
        return None


def main() -> None:
    settings = get_settings()
    configure_logging(settings.app.log_level)

    suite_dir = Path(__file__).resolve().parent
    query_files = [suite_dir / "test_queries.json"]
    if os.getenv("EVAL_SUITE", "single").lower() == "all":
        query_files.extend([suite_dir / "test_queries_en.json", suite_dir / "test_queries_hk_mixed.json"])
    test_queries: list[dict[str, Any]] = []
    for queries_path in query_files:
        if not queries_path.exists():
            logger.error("Test queries file not found: %s", queries_path)
            raise SystemExit(1)
        test_queries.extend(load_test_queries(queries_path))
    agent_graph, pipeline, embedding_client = build_agent_stack(settings)

    records = run_generation_eval(test_queries, agent_graph)
    if not records:
        logger.error("No annotated queries were available for generation evaluation")
        raise SystemExit(1)

    try:
        scores = _evaluate_ragas(records, settings, embedding_client)
        scores.update(_safety_scores(records, settings.inference.nli_entailment_threshold))
    except Exception as exc:  # noqa: BLE001
        logger.error("RAGAs evaluation failed: %s", exc)
        raise SystemExit(1)

    print_report(scores, len(records))

    output_path = write_results(Path(__file__).resolve().parents[2] / "results", scores, len(records))
    if output_path is not None:
        logger.info("Wrote RAGAs evaluation results to %s", output_path)


if __name__ == "__main__":
    main()
