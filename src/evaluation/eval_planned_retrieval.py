from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from config.settings import configure_logging, get_settings
from src.evaluation.eval_retrieval import build_metrics, load_test_queries
from src.indexing import BGEM3EmbeddingClient, MilvusHybridStore
from src.retrieval.query_pipeline import (
    ComplianceRetrievalPipeline,
    ConfigurableLLMProvider,
    CrossEncoderRerankerClient,
    RetrievalRequest,
)

logger = logging.getLogger(__name__)
_SUITE_FILES = {
    "en": "test_queries_en_v2.json",
    "zh": "test_queries_v2.json",
    "hk_mixed": "test_queries_hk_mixed_v2.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the planned retrieval pipeline by stage.")
    parser.add_argument("--suite", choices=["all", "single"], default="all")
    parser.add_argument("--queries-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/planned_retrieval_v2_metrics.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Evaluate multi-query RRF and parent backfill without reranking or HyDE.",
    )
    return parser.parse_args(argv)


def load_suite(args: argparse.Namespace) -> list[dict[str, Any]]:
    suite_dir = Path(__file__).resolve().parent
    records: list[dict[str, Any]] = []
    if args.suite == "single":
        if args.queries_file is None:
            raise ValueError("--queries-file is required when --suite single is used")
        for record in load_test_queries(args.queries_file):
            records.append({**record, "_suite": args.queries_file.stem})
    else:
        for suite_name, filename in _SUITE_FILES.items():
            for record in load_test_queries(suite_dir / filename):
                records.append({**record, "_suite": suite_name})
    return records[: args.limit] if args.limit > 0 else records


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _stage_metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    evaluation_rows = [
        {
            "ground_truth_chunk_ids": record["ground_truth_chunk_ids"],
            "retrieved_chunk_ids": record[field],
        }
        for record in records
    ]
    overall = build_metrics(evaluation_rows)
    by_suite: dict[str, dict[str, float]] = {}
    for suite_name in sorted({str(record["suite"]) for record in records}):
        suite_rows = [row for row, record in zip(evaluation_rows, records) if record["suite"] == suite_name]
        if suite_rows:
            by_suite[suite_name] = build_metrics(suite_rows)
    return {"overall": overall, "by_suite": by_suite}


async def evaluate(records: list[dict[str, Any]], *, enable_reranker: bool = True) -> dict[str, Any]:
    settings = get_settings()
    if enable_reranker and not settings.inference.reranker_service_url:
        raise RuntimeError("RERANKER_SERVICE_URL is required for planned retrieval evaluation")
    if not enable_reranker:
        settings = replace(
            settings,
            retrieval=replace(
                settings.retrieval,
                hyde_score_threshold=-1.0,
                hyde_margin_threshold=-1.0,
            ),
        )

    planner = ConfigurableLLMProvider(settings)
    pipeline = ComplianceRetrievalPipeline(
        store=MilvusHybridStore(settings),
        embedding_client=BGEM3EmbeddingClient(settings.inference.embedding_model_name),
        reranker=(
            CrossEncoderRerankerClient(settings.inference.reranker_model_name)
            if enable_reranker
            else None
        ),
        llm_provider=planner,
        settings=settings,
    )

    evaluated: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    latencies: list[float] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for index, record in enumerate(records, start=1):
        query = str(record.get("query", "")).strip()
        logger.info("[%d/%d] Planned retrieval: %s", index, len(records), query[:80])
        planner.reset_token_usage()
        started = time.perf_counter()
        try:
            result = await pipeline.retrieve(RetrievalRequest(user_query=query))
        except Exception as exc:  # noqa: BLE001
            logger.error("Planned retrieval failed for '%s': %s", query[:60], exc)
            errors.append({"query": query, "error": str(exc)})
            evaluated.append(
                {
                    "query": query,
                    "suite": record["_suite"],
                    "ground_truth_chunk_ids": record.get("ground_truth_chunk_ids", []),
                    "expanded_rrf_ids": [],
                    "expanded_reranked_ids": [],
                    "adaptive_final_ids": [],
                    "hyde_triggered": False,
                    "expanded_query_count": 0,
                    "parent_context_count": 0,
                    "final_chunk_count": 0,
                    "latency_seconds": time.perf_counter() - started,
                }
            )
            continue

        latency = time.perf_counter() - started
        latencies.append(latency)
        usage = planner.token_usage
        for key in total_usage:
            total_usage[key] += int(usage.get(key, 0))

        evaluated.append(
            {
                "query": query,
                "suite": record["_suite"],
                "ground_truth_chunk_ids": record.get("ground_truth_chunk_ids", []),
                "expanded_rrf_ids": [chunk.chunk_id for chunk in result.initial_retrieved_chunks],
                "expanded_reranked_ids": [chunk.chunk_id for chunk in result.initial_reranked_chunks],
                "adaptive_final_ids": [chunk.chunk_id for chunk in result.reranked_chunks],
                "hyde_triggered": bool(result.hyde_hypothesis),
                "expanded_query_count": len(result.expanded_queries),
                "expanded_queries": result.expanded_queries,
                "parent_context_count": sum(chunk.chunk_type == "parent_context" for chunk in result.reranked_chunks),
                "final_chunk_count": len(result.reranked_chunks),
                "latency_seconds": latency,
                "planner_usage": usage,
                "audit_decision": asdict(result.audit_decision),
            }
        )

    final_chunks = sum(record["final_chunk_count"] for record in evaluated)
    parent_contexts = sum(record["parent_context_count"] for record in evaluated)
    diagnostics = {
        "query_count": len(evaluated),
        "error_count": len(errors),
        "hyde_trigger_count": sum(record["hyde_triggered"] for record in evaluated),
        "hyde_trigger_rate": (
            sum(record["hyde_triggered"] for record in evaluated) / len(evaluated)
            if evaluated
            else 0.0
        ),
        "average_expanded_query_count": statistics.fmean(
            record["expanded_query_count"] for record in evaluated
        ) if evaluated else 0.0,
        "parent_backfill_coverage": parent_contexts / final_chunks if final_chunks else 0.0,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
        "planner_token_usage": total_usage,
    }
    stages = {
        "expanded_rrf": _stage_metrics(evaluated, "expanded_rrf_ids"),
    }
    if enable_reranker:
        stages["expanded_reranked"] = _stage_metrics(evaluated, "expanded_reranked_ids")
        stages["adaptive_hyde_reranked_parent"] = _stage_metrics(evaluated, "adaptive_final_ids")
    else:
        stages["expanded_top8_parent"] = _stage_metrics(evaluated, "adaptive_final_ids")

    return {
        "metadata": {
            "collection": settings.milvus.collection_name,
            "planner_provider": settings.llm_roles.planner.provider,
            "planner_model": settings.llm_roles.planner.model,
            "embedding_model": settings.inference.embedding_model_name,
            "reranker_model": settings.inference.reranker_model_name,
            "reranker_enabled": enable_reranker,
            "hyde_score_threshold": settings.retrieval.hyde_score_threshold,
            "hyde_margin_threshold": settings.retrieval.hyde_margin_threshold,
            "parent_backfill": settings.retrieval.parent_backfill,
        },
        "stages": stages,
        "diagnostics": diagnostics,
        "errors": errors,
        "records": evaluated,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.app.log_level)
    records = load_suite(args)
    report = asyncio.run(evaluate(records, enable_reranker=not args.no_reranker))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stages": report["stages"], "diagnostics": report["diagnostics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
