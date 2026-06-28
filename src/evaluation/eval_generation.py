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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import configure_logging, get_anthropic_client, get_settings
from src.agent import ComplianceAgentGraph
from src.indexing import BGEM3EmbeddingClient, MilvusHybridStore
from src.retrieval import ComplianceRetrievalPipeline, CrossEncoderRerankerClient, RetrievalRequest

logger = logging.getLogger(__name__)

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

        def get_temperature(self) -> float:
            return 0.0

        def _request_text(
            self,
            prompt: Any,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
        ) -> str:
            request: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": (
                            "You are a strict evaluator for a regulated-finance retrieval benchmark. "
                            "Follow the prompt exactly and return only the requested output."
                        ),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": _prompt_to_text(prompt)}],
                "output_config": {"effort": self.effort},
            }
            if temperature is not None:
                request["temperature"] = temperature
            if stop:
                request["stop_sequences"] = stop

            response = self._client.messages.create(**request)
            return _response_text(getattr(response, "content", []))

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> LLMResult:
            text = self._request_text(prompt, temperature=temperature, stop=stop)
            generations = [Generation(text=text) for _ in range(max(1, n))]
            return LLMResult(generations=[generations], llm_output={"model": self.model_name})

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> LLMResult:
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n,
                temperature,
                stop,
                callbacks,
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

        def get_temperature(self) -> float:
            return 0.0

        def _request_text(
            self,
            prompt: Any,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
        ) -> str:
            request: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": (
                            "You are a strict evaluator for a regulated-finance retrieval benchmark. "
                            "Follow the prompt exactly and return only the requested output."
                        ),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": _prompt_to_text(prompt)}],
                "output_config": {"effort": self.effort},
            }
            if temperature is not None:
                request["temperature"] = temperature
            if stop:
                request["stop_sequences"] = stop

            response = self._client.messages.create(**request)
            return _response_text(getattr(response, "content", []))

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 1e-8,
            stop: list[str] | None = None,
            callbacks: Any = None,
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
        ) -> Any:
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n,
                temperature,
                stop,
                callbacks,
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
            return self._embed_many([text])[0]

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
            return self._embed_many([text])[0]

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
    try:
        reranker = CrossEncoderRerankerClient(settings.inference.reranker_model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranker unavailable, using dense-only retrieval mode: %s", exc)

    pipeline = ComplianceRetrievalPipeline(
        store=store,
        embedding_client=embedding_client,
        reranker=reranker,
        settings=settings,
    )
    agent_graph = ComplianceAgentGraph(retrieval_pipeline=pipeline, settings=settings)
    return agent_graph, pipeline, embedding_client


def _extract_contexts(retrieval_result: Any) -> list[str]:
    contexts: list[str] = []
    for chunk in list(getattr(retrieval_result, "reranked_chunks", []) or []):
        text = str(getattr(chunk, "text", "")).strip()
        if text:
            contexts.append(text)
    return contexts


def _retrieve_contexts(pipeline: ComplianceRetrievalPipeline, query: str) -> list[str]:
    retrieval_result = _run_coroutine(pipeline.retrieve(RetrievalRequest(user_query=query)))
    return _extract_contexts(retrieval_result)


def run_generation_eval(
    test_queries: list[dict[str, Any]],
    agent_graph: ComplianceAgentGraph,
    pipeline: ComplianceRetrievalPipeline,
) -> list[dict[str, Any]]:
    """Run the graph against annotated queries and collect evaluation records."""
    records: list[dict[str, Any]] = []
    annotated_queries = [q for q in test_queries if str(q.get("ground_truth_answer", "")).strip()]

    for index, item in enumerate(annotated_queries, start=1):
        query = str(item.get("query", "")).strip()
        if not query:
            continue

        logger.info("[%d/%d] Evaluating generation: %s", index, len(annotated_queries), query[:80])

        try:
            contexts = _retrieve_contexts(pipeline, query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval failed for query '%s': %s", query[:60], exc)
            contexts = []

        try:
            result = agent_graph.invoke({"user_query": query})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent evaluation failed for query '%s': %s", query[:60], exc)
            continue

        records.append(
            {
                "question": query,
                "answer": str(getattr(result, "answer", "") or ""),
                "contexts": contexts,
                "ground_truth": str(item.get("ground_truth_answer", "") or "").strip(),
            }
        )

    return records


def _format_bar(value: float, width: int = 24) -> str:
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
        print(f"\n  {label:17} {score:.4f}  [{_format_bar(score)}]")

    print("\n" + "=" * 72)


def _extract_metric_score(result: Any, metric_name: str) -> float:
    scores = getattr(result, "scores", None)
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

    llm_client = get_anthropic_client(settings)
    llm = AnthropicRagasLLM(
        llm_client,
        settings.llm.model,
        settings.llm.max_tokens,
        settings.llm.effort,
    )
    embeddings = BGEM3RagasEmbeddings(embedding_client)

    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextRecall()]
    result = evaluate(
        hf_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    return {
        "faithfulness": _extract_metric_score(result, "faithfulness"),
        "answer_relevancy": _extract_metric_score(result, "answer_relevancy"),
        "context_recall": _extract_metric_score(result, "context_recall"),
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

    queries_path = Path(__file__).resolve().parent / "test_queries.json"
    if not queries_path.exists():
        logger.error("Test queries file not found: %s", queries_path)
        raise SystemExit(1)

    test_queries = load_test_queries(queries_path)
    agent_graph, pipeline, embedding_client = build_agent_stack(settings)

    records = run_generation_eval(test_queries, agent_graph, pipeline)
    if not records:
        logger.error("No annotated queries were available for generation evaluation")
        raise SystemExit(1)

    try:
        scores = _evaluate_ragas(records, settings, embedding_client)
    except Exception as exc:  # noqa: BLE001
        logger.error("RAGAs evaluation failed: %s", exc)
        raise SystemExit(1)

    print_report(scores, len(records))

    output_path = write_results(Path(__file__).resolve().parents[2] / "results", scores, len(records))
    if output_path is not None:
        logger.info("Wrote RAGAs evaluation results to %s", output_path)


if __name__ == "__main__":
    main()
