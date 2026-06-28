# ARCHIVED — 2024 实验版本，当前入口为 app.py

import importlib.util
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DENSE_TOP_K = 10
DEFAULT_TOP_K = 5
VERIFICATION_QUERIES = [
    "What are the customer due diligence requirements?",
    "Section 4.1 AML/CFT obligations",
    "risk assessment procedures for money laundering",
]

logger = logging.getLogger(__name__)
_dense_module: Optional[Any] = None
_cross_encoder: Optional[CrossEncoder] = None


def configure_logging() -> None:
    """配置标准日志，输出时间戳、等级和消息内容。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_dense_module() -> Any:
    """通过 importlib 复用 5_hybrid_retrieval.py 中的 dense 检索逻辑。"""
    global _dense_module

    if _dense_module is None:
        module_path = Path(__file__).resolve().parent / "5_hybrid_retrieval.py"
        spec = importlib.util.spec_from_file_location("hybrid_retrieval_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load dense retrieval module from: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _dense_module = module

    return _dense_module


def get_cross_encoder() -> CrossEncoder:
    """延迟初始化 Cross-Encoder 模型。"""
    global _cross_encoder

    if _cross_encoder is None:
        logger.info("Initializing CrossEncoder model: %s", CROSS_ENCODER_MODEL_NAME)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device="cpu")

    return _cross_encoder


def dense_candidate_retrieve(
    query: str,
    db_dir: Path,
    candidate_k: int = DEFAULT_DENSE_TOP_K,
) -> List[Dict[str, Any]]:
    """调用既有 dense_retrieve，获取父节点候选列表。"""
    dense_module = load_dense_module()
    dense_retrieve = getattr(dense_module, "dense_retrieve", None)
    if dense_retrieve is None:
        raise AttributeError("5_hybrid_retrieval.py does not define dense_retrieve")

    return dense_retrieve(query=query, db_dir=db_dir, top_k=candidate_k)


def rerank_candidates(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """使用 Cross-Encoder 对 dense 候选进行重排。"""
    if not candidates:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [(query, candidate.get("text", "")) for candidate in candidates]
    scores = cross_encoder.predict(pairs)

    reranked_candidates: List[Dict[str, Any]] = []
    for dense_rank, (candidate, cross_score) in enumerate(zip(candidates, scores), start=1):
        candidate_with_scores = dict(candidate)
        candidate_with_scores["dense_rank"] = dense_rank
        candidate_with_scores["dense_score"] = float(candidate.get("score", 0.0))
        candidate_with_scores["cross_encoder_score"] = float(cross_score)
        reranked_candidates.append(candidate_with_scores)

    return sorted(
        reranked_candidates,
        key=lambda item: item["cross_encoder_score"],
        reverse=True,
    )


def format_reranked_results(reranked_candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """将重排后的父节点结果映射回标准 evidence 格式。"""
    formatted_results = []

    for index, item in enumerate(reranked_candidates[:top_k], start=1):
        formatted_results.append(
            {
                "label": f"E{index}",
                "source_file": item["source_file"],
                "page": item["page"],
                "page_display": item["page_display"],
                "score": round(float(item["cross_encoder_score"]), 3),
                "text": item["text"],
            }
        )

    return formatted_results


def cross_encoder_retrieve(query: str, db_dir: Path, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """执行 Dense -> Cross-Encoder 两阶段检索，并返回最终 evidence 列表。"""
    dense_candidates = dense_candidate_retrieve(query=query, db_dir=db_dir, candidate_k=DEFAULT_DENSE_TOP_K)
    reranked_candidates = rerank_candidates(query=query, candidates=dense_candidates)
    return format_reranked_results(reranked_candidates=reranked_candidates, top_k=top_k)


def run_cross_encoder_pipeline(
    query: str,
    db_dir: Path,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """运行完整两阶段检索，返回结果、排名变化与耗时信息。"""
    stage1_start = time.perf_counter()
    dense_candidates = dense_candidate_retrieve(query=query, db_dir=db_dir, candidate_k=DEFAULT_DENSE_TOP_K)
    stage1_elapsed = time.perf_counter() - stage1_start

    stage2_start = time.perf_counter()
    reranked_candidates = rerank_candidates(query=query, candidates=dense_candidates)
    stage2_elapsed = time.perf_counter() - stage2_start

    formatted_results = format_reranked_results(reranked_candidates=reranked_candidates, top_k=top_k)

    reranked_with_positions: List[Dict[str, Any]] = []
    for reranked_rank, item in enumerate(reranked_candidates, start=1):
        reranked_with_positions.append(
            {
                "source_file": item["source_file"],
                "page": item["page"],
                "page_display": item["page_display"],
                "dense_rank": item["dense_rank"],
                "dense_score": round(float(item["dense_score"]), 3),
                "reranked_rank": reranked_rank,
                "cross_encoder_score": round(float(item["cross_encoder_score"]), 3),
                "text": item["text"],
            }
        )

    return {
        "query": query,
        "results": formatted_results,
        "dense_candidates": dense_candidates,
        "reranked_candidates": reranked_with_positions,
        "stage1_elapsed": stage1_elapsed,
        "stage2_elapsed": stage2_elapsed,
        "total_elapsed": stage1_elapsed + stage2_elapsed,
    }


def preview_text(text: str, limit: int = 120) -> str:
    """生成单行文本预览。"""
    preview = (text or "").replace("\n", " ").strip()
    if len(preview) > limit:
        preview = f"{preview[:limit]}..."
    return preview


def print_query_report(summary: Dict[str, Any], top_n: int = 3) -> None:
    """打印单个查询的 top 结果、耗时与重排前后对比。"""
    separator = "=" * 100
    dense_rank_map = {
        candidate["dense_rank"]: candidate
        for candidate in summary["reranked_candidates"]
    }

    print(separator)
    print(f"QUERY: {summary['query']}")
    print(separator)
    print("Top Re-ranked Results")
    print("-" * 100)

    for result in summary["results"][:top_n]:
        matching_candidate = next(
            (
                item
                for item in summary["reranked_candidates"]
                if item["page"] == result["page"]
                and item["source_file"] == result["source_file"]
                and item["text"] == result["text"]
            ),
            None,
        )
        dense_rank_display = matching_candidate["dense_rank"] if matching_candidate else "N/A"
        print(
            f"{result['label']} | page={result['page_display']} | score={result['score']} | "
            f"dense_rank={dense_rank_display} | {preview_text(result['text'])}"
        )

    print("-" * 100)
    print(
        f"Stage 1 (dense) : {summary['stage1_elapsed']:.4f}s | "
        f"Stage 2 (rerank) : {summary['stage2_elapsed']:.4f}s | "
        f"Total : {summary['total_elapsed']:.4f}s"
    )
    print("-" * 100)
    print("Before/After Ranking")
    print("-" * 100)

    for item in summary["reranked_candidates"][:top_n]:
        print(
            f"dense#{item['dense_rank']} -> reranked#{item['reranked_rank']} | "
            f"page={item['page_display']} | ce_score={item['cross_encoder_score']} | "
            f"dense_score={item['dense_score']}"
        )

    print(separator)


def resolve_db_dir() -> Path:
    """解析默认 ChromaDB 目录。"""
    return (Path(__file__).resolve().parent / "chroma_db").resolve()


def main() -> int:
    """执行三条验证查询，展示 Cross-Encoder 重排效果。"""
    configure_logging()

    try:
        db_dir = resolve_db_dir()
        logger.info("Resolved ChromaDB directory: %s", db_dir)

        for query in VERIFICATION_QUERIES:
            summary = run_cross_encoder_pipeline(query=query, db_dir=db_dir, top_k=3)
            print_query_report(summary=summary, top_n=3)

        return 0
    except Exception as exc:
        logger.exception("Unexpected failure while running cross-encoder retrieval: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
