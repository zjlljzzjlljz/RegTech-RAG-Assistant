import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anthropic
from dotenv import load_dotenv

HYDE_MODEL_NAME = "claude-fable-5"
DEFAULT_CANDIDATE_K = 10
DEFAULT_TOP_K = 5
VERIFICATION_QUERIES = [
    "What are the customer due diligence requirements?",
    "Section 4.1 AML/CFT obligations",
    "risk assessment procedures for money laundering",
]
HYDE_SYSTEM_PROMPT = (
    "You are a regulatory document retrieval assistant. Given a user's question "
    "about HKMA AML/CFT compliance, write a hypothetical paragraph that a "
    "compliance guideline document would contain to answer this question. "
    "Use formal regulatory terminology (e.g. 'Risk-Based Approach', 'Customer "
    "Due Diligence measures', 'AML/CFT Systems', 'Schedule 2 requirements'). "
    "Write 3-5 sentences. Do NOT answer the question — write what the IDEAL "
    "SOURCE DOCUMENT would say."
)

logger = logging.getLogger(__name__)
_dense_module: Optional[Any] = None
_cross_encoder_module: Optional[Any] = None
_environment_loaded = False


def configure_logging() -> None:
    """配置标准日志，输出时间戳、等级和消息内容。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_base_dir() -> Path:
    """解析项目根目录。"""
    return Path(__file__).resolve().parent


def resolve_db_dir() -> Path:
    """解析默认 ChromaDB 目录。"""
    return (resolve_base_dir() / "chroma_db").resolve()


def ensure_environment_loaded() -> None:
    """从项目根目录加载 .env，确保 HyDE 生成函数可独立工作。"""
    global _environment_loaded

    if _environment_loaded:
        return

    env_path = resolve_base_dir() / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info("Environment variables loaded from: %s", env_path)
    _environment_loaded = True


def load_module(module_path: Path, module_name: str) -> Any:
    """使用 importlib 从脚本路径加载模块。"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dense_module() -> Any:
    """加载 5_hybrid_retrieval.py，复用 dense 检索逻辑。"""
    global _dense_module

    if _dense_module is None:
        module_path = resolve_base_dir() / "5_hybrid_retrieval.py"
        _dense_module = load_module(module_path=module_path, module_name="hybrid_retrieval_module")

    return _dense_module


def load_cross_encoder_module() -> Any:
    """加载 6_cross_encoder_retrieval.py，复用 Cross-Encoder 重排逻辑。"""
    global _cross_encoder_module

    if _cross_encoder_module is None:
        module_path = resolve_base_dir() / "6_cross_encoder_retrieval.py"
        _cross_encoder_module = load_module(module_path=module_path, module_name="cross_encoder_retrieval_module")

    return _cross_encoder_module


def get_dense_retrieve():
    """返回 dense_retrieve 函数引用。"""
    module = load_dense_module()
    dense_retrieve = getattr(module, "dense_retrieve", None)
    if dense_retrieve is None:
        raise AttributeError("5_hybrid_retrieval.py does not define dense_retrieve")
    return dense_retrieve


def get_cross_encoder():
    """返回 Cross-Encoder 模型实例。"""
    module = load_cross_encoder_module()
    get_model = getattr(module, "get_cross_encoder", None)
    if get_model is None:
        raise AttributeError("6_cross_encoder_retrieval.py does not define get_cross_encoder")
    return get_model()


def get_rerank_candidates():
    """返回候选重排函数。"""
    module = load_cross_encoder_module()
    rerank_candidates = getattr(module, "rerank_candidates", None)
    if rerank_candidates is None:
        raise AttributeError("6_cross_encoder_retrieval.py does not define rerank_candidates")
    return rerank_candidates


def get_anthropic_client() -> anthropic.Anthropic:
    """按照 3_app.py 的环境规则创建 Anthropic 客户端。"""
    ensure_environment_loaded()
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    auth_token = (os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()

    if base_url and auth_token == "PROXY_MANAGED":
        logger.info("Detected cc-switch local proxy at %s", base_url)
        return anthropic.Anthropic(auth_token=auth_token, base_url=base_url)

    if base_url:
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Please add it to your .env file when using a custom ANTHROPIC_BASE_URL."
            )
        return anthropic.Anthropic(api_key=api_key, base_url=base_url)

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Please add it to your .env file.")

    return anthropic.Anthropic(api_key=api_key)


def extract_text_from_message(response: Any) -> str:
    """从 Anthropic Messages 响应中提取文本。"""
    texts: List[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            block_text = getattr(block, "text", "")
            if block_text:
                texts.append(block_text)
    return "\n".join(texts).strip()


def generate_hyde_query(original_query: str) -> str:
    """使用 Claude 生成假设性法规段落；失败时回退为原始查询。"""
    try:
        client = get_anthropic_client()
        response = client.messages.create(
            model=HYDE_MODEL_NAME,
            max_tokens=200,
            temperature=0,
            system=HYDE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": original_query,
                }
            ],
        )
        hyde_query = extract_text_from_message(response)
        if not hyde_query:
            logger.warning("HyDE generation returned empty content; falling back to original query.")
            return original_query
        return hyde_query
    except Exception as exc:
        logger.warning("HyDE generation failed; falling back to original query: %s", exc)
        return original_query


def preview_text(text: str, limit: int = 24) -> str:
    """生成单行短预览。"""
    normalized = (text or "").replace("\n", " ").strip()
    if len(normalized) > limit:
        normalized = f"{normalized[:limit]}..."
    return normalized


def format_dense_results(dense_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """将 dense 检索结果映射为标准 evidence 结构。"""
    formatted_results: List[Dict[str, Any]] = []

    for index, item in enumerate(dense_results[:top_k], start=1):
        formatted_results.append(
            {
                "label": f"E{index}",
                "source_file": item["source_file"],
                "page": item["page"],
                "page_display": item["page_display"],
                "score": round(float(item.get("score", 0.0)), 3),
                "text": item["text"],
            }
        )

    return formatted_results


def format_reranked_results(
    reranked_candidates: List[Dict[str, Any]],
    top_k: int,
    hyde_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """将 Cross-Encoder 重排结果映射为标准 evidence 结构。"""
    formatted_results: List[Dict[str, Any]] = []

    for index, item in enumerate(reranked_candidates[:top_k], start=1):
        result = {
            "label": f"E{index}",
            "source_file": item["source_file"],
            "page": item["page"],
            "page_display": item["page_display"],
            "score": round(float(item.get("cross_encoder_score", item.get("score", 0.0))), 3),
            "text": item["text"],
        }
        if hyde_query is not None:
            result["hyde_query"] = hyde_query
        formatted_results.append(result)

    return formatted_results


def evaluate_pipeline(query: str, results: List[Dict[str, Any]]) -> float:
    """用 Cross-Encoder 对 top-k 结果做统一相关性打分，便于自动选胜者。"""
    if not results:
        return float("-inf")

    try:
        cross_encoder = get_cross_encoder()
        pairs = [(query, result.get("text", "")) for result in results]
        scores = cross_encoder.predict(pairs)
        return float(sum(float(score) for score in scores))
    except Exception as exc:
        logger.warning("Pipeline evaluation fallback triggered: %s", exc)
        return float(sum(float(result.get("score", 0.0)) for result in results))


def render_result_cell(results: List[Dict[str, Any]], index: int) -> str:
    """将某个排名结果渲染为表格单元格。"""
    if index >= len(results):
        return "-"

    result = results[index]
    return f"p.{result['page_display']} {preview_text(result['text'])}"


def print_comparison_table(rows: List[Tuple[str, List[str]]]) -> None:
    """打印三路检索对比表。"""
    headers = ["Pipeline", "Rank #1", "Rank #2", "Rank #3"]
    widths = [len(header) for header in headers]

    for row in rows:
        widths[0] = max(widths[0], len(row[0]))
        for index, cell in enumerate(row[1], start=1):
            widths[index] = max(widths[index], len(cell))

    def border(left: str, fill: str, junction: str, right: str) -> str:
        return left + junction.join(fill * (width + 2) for width in widths) + right

    def render_row(cells: Sequence[str]) -> str:
        return "│ " + " │ ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)) + " │"

    print(border("┌", "─", "┬", "┐"))
    print(render_row(headers))
    print(border("├", "─", "┼", "┤"))
    for pipeline_name, cells in rows:
        print(render_row([pipeline_name] + cells))
    print(border("└", "─", "┴", "┘"))


def run_dense_pipeline(query: str, db_dir: Path, top_k: int = 3) -> Dict[str, Any]:
    """运行 baseline dense 检索。"""
    dense_retrieve = get_dense_retrieve()

    dense_start = time.perf_counter()
    dense_candidates = dense_retrieve(query=query, db_dir=db_dir, top_k=top_k)
    dense_elapsed = time.perf_counter() - dense_start

    results = format_dense_results(dense_candidates, top_k=top_k)
    return {
        "name": "Dense",
        "results": results,
        "query_generation_elapsed": 0.0,
        "dense_elapsed": dense_elapsed,
        "rerank_elapsed": 0.0,
        "total_elapsed": dense_elapsed,
    }


def run_cross_encoder_only_pipeline(query: str, db_dir: Path, top_k: int = 3) -> Dict[str, Any]:
    """运行 Dense -> Cross-Encoder 重排管线；失败时回退为 dense-only。"""
    dense_retrieve = get_dense_retrieve()
    rerank_candidates = get_rerank_candidates()

    dense_start = time.perf_counter()
    dense_candidates = dense_retrieve(query=query, db_dir=db_dir, top_k=DEFAULT_CANDIDATE_K)
    dense_elapsed = time.perf_counter() - dense_start

    rerank_start = time.perf_counter()
    fallback_used = False
    try:
        reranked_candidates = rerank_candidates(query=query, candidates=dense_candidates)
        results = format_reranked_results(reranked_candidates, top_k=top_k)
    except Exception as exc:
        logger.warning("Cross-encoder rerank failed; falling back to dense-only results: %s", exc)
        fallback_used = True
        reranked_candidates = []
        results = format_dense_results(dense_candidates, top_k=top_k)
    rerank_elapsed = time.perf_counter() - rerank_start

    return {
        "name": "X-Encode",
        "results": results,
        "query_generation_elapsed": 0.0,
        "dense_elapsed": dense_elapsed,
        "rerank_elapsed": rerank_elapsed,
        "total_elapsed": dense_elapsed + rerank_elapsed,
        "fallback_used": fallback_used,
        "dense_candidates": dense_candidates,
        "reranked_candidates": reranked_candidates,
    }


def hyde_retrieve(query: str, db_dir: Path, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """执行 HyDE -> Dense -> Cross-Encoder 检索；失败时优雅降级。"""
    summary = run_hyde_pipeline(query=query, db_dir=db_dir, top_k=top_k)
    return summary["results"]


def run_hyde_pipeline(query: str, db_dir: Path, top_k: int = 3) -> Dict[str, Any]:
    """运行 HyDE + Cross-Encoder 管线，并返回完整耗时与降级信息。"""
    dense_retrieve = get_dense_retrieve()
    rerank_candidates = get_rerank_candidates()

    generation_start = time.perf_counter()
    hyde_query = generate_hyde_query(query)
    query_generation_elapsed = time.perf_counter() - generation_start

    dense_start = time.perf_counter()
    dense_candidates = dense_retrieve(query=hyde_query, db_dir=db_dir, top_k=DEFAULT_CANDIDATE_K)
    dense_elapsed = time.perf_counter() - dense_start

    rerank_start = time.perf_counter()
    fallback_used = False
    try:
        reranked_candidates = rerank_candidates(query=query, candidates=dense_candidates)
        results = format_reranked_results(reranked_candidates, top_k=top_k, hyde_query=hyde_query)
    except Exception as exc:
        logger.warning("HyDE cross-encoder rerank failed; falling back to dense-only results: %s", exc)
        fallback_used = True
        reranked_candidates = []
        results = format_dense_results(dense_candidates, top_k=top_k)
        for result in results:
            result["hyde_query"] = hyde_query
    rerank_elapsed = time.perf_counter() - rerank_start

    return {
        "name": "HyDE+XE",
        "hyde_query": hyde_query,
        "results": results,
        "query_generation_elapsed": query_generation_elapsed,
        "dense_elapsed": dense_elapsed,
        "rerank_elapsed": rerank_elapsed,
        "total_elapsed": query_generation_elapsed + dense_elapsed + rerank_elapsed,
        "fallback_used": fallback_used,
        "dense_candidates": dense_candidates,
        "reranked_candidates": reranked_candidates,
    }


def determine_winner(query: str, pipeline_summaries: List[Dict[str, Any]]) -> str:
    """根据 top-3 聚合相关性分数自动选出当前查询的优胜管线。"""
    best_name = "Dense"
    best_score = float("-inf")

    for summary in pipeline_summaries:
        aggregate_score = evaluate_pipeline(query=query, results=summary["results"])
        summary["aggregate_score"] = round(aggregate_score, 3)
        if aggregate_score > best_score:
            best_score = aggregate_score
            best_name = summary["name"]

    return best_name


def print_timing_report(pipeline_summaries: List[Dict[str, Any]]) -> None:
    """打印三路管线耗时与相关性聚合分数。"""
    for summary in pipeline_summaries:
        print(
            f"{summary['name']}: gen={summary['query_generation_elapsed']:.4f}s | "
            f"dense={summary['dense_elapsed']:.4f}s | "
            f"rerank={summary['rerank_elapsed']:.4f}s | "
            f"total={summary['total_elapsed']:.4f}s | "
            f"judge_score={summary.get('aggregate_score', 0.0):.3f}"
        )


def print_query_benchmark(query: str, db_dir: Path) -> None:
    """打印单个查询的三路检索基准对比。"""
    separator = "=" * 100
    dense_summary = run_dense_pipeline(query=query, db_dir=db_dir, top_k=3)
    cross_encoder_summary = run_cross_encoder_only_pipeline(query=query, db_dir=db_dir, top_k=3)
    hyde_summary = run_hyde_pipeline(query=query, db_dir=db_dir, top_k=3)

    pipeline_summaries = [dense_summary, cross_encoder_summary, hyde_summary]
    winner = determine_winner(query=query, pipeline_summaries=pipeline_summaries)

    print(separator)
    print(f'QUERY: "{query}"')
    print(separator)
    print("HyDE expanded query:")
    print(hyde_summary["hyde_query"])
    print("-" * 100)

    rows = []
    for summary in pipeline_summaries:
        cells = [render_result_cell(summary["results"], index) for index in range(3)]
        rows.append((summary["name"], cells))

    print_comparison_table(rows)
    print(f"Winner: {winner} (by top-3 aggregate cross-encoder relevance to the original query)")
    print("-" * 100)
    print_timing_report(pipeline_summaries)
    print(separator)


def main() -> int:
    """运行 Dense / Cross-Encoder / HyDE+XE 三路对比基准。"""
    configure_logging()

    try:
        ensure_environment_loaded()
        db_dir = resolve_db_dir()
        logger.info("Resolved ChromaDB directory: %s", db_dir)

        for query in VERIFICATION_QUERIES:
            print_query_benchmark(query=query, db_dir=db_dir)

        return 0
    except Exception as exc:
        logger.exception("Unexpected failure while running HyDE retrieval benchmark: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
