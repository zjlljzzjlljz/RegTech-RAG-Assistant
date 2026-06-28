# ARCHIVED — 2024 实验版本，当前入口为 app.py

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# SQLite3 兼容性猴子补丁
# -----------------------------
# 必须放在文件最顶部，并且早于 chromadb 相关导入。
# 目的：在本地系统 sqlite3 版本过低时，优先尝试切换到 pysqlite3。
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import logging
import sqlite3

import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MIN_SQLITE_VERSION = (3, 35, 0)
COLLECTION_NAME = "regtech_parent_child_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60
DEFAULT_DENSE_TOP_K = 10
DEFAULT_SPARSE_TOP_K = 10

logger = logging.getLogger(__name__)
_embedding_model: Optional[SentenceTransformer] = None


def _parse_sqlite_version(version: str) -> tuple[int, int, int]:
    """将 sqlite3 版本字符串解析为可比较的三元组。"""
    parsed_parts = []
    for part in version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parsed_parts.append(int(digits))
        if len(parsed_parts) == 3:
            break

    while len(parsed_parts) < 3:
        parsed_parts.append(0)

    return tuple(parsed_parts[:3])


ACTIVE_SQLITE_VERSION = sqlite3.sqlite_version
ACTIVE_SQLITE_VERSION_TUPLE = _parse_sqlite_version(ACTIVE_SQLITE_VERSION)
ACTIVE_SQLITE_BACKEND = getattr(sqlite3, "__file__", sqlite3.__name__)

if ACTIVE_SQLITE_VERSION_TUPLE < MIN_SQLITE_VERSION:
    raise RuntimeError(
        "Unsupported sqlite3 runtime detected before Chroma initialization. "
        f"Active backend: {ACTIVE_SQLITE_BACKEND}; version: {ACTIVE_SQLITE_VERSION}. "
        "Chroma requires sqlite3 >= 3.35.0. Install pysqlite3-binary into the active interpreter, "
        "or run this script with a Python build linked against a newer SQLite runtime."
    )


def configure_logging() -> None:
    """配置标准日志，输出时间戳、等级和消息内容。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_sqlite_runtime() -> None:
    """记录当前 SQLite 后端与版本，便于排查 Chroma 运行环境问题。"""
    logger.info("Active sqlite backend: %s", ACTIVE_SQLITE_BACKEND)
    logger.info("Active sqlite version: %s", ACTIVE_SQLITE_VERSION)


def load_environment(base_dir: Path) -> None:
    """从项目根目录加载 .env 文件。"""
    env_path = base_dir / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info("Environment variables loaded from: %s", env_path)


def resolve_paths() -> Tuple[Path, Path]:
    """统一解析项目根目录与 ChromaDB 持久化目录。"""
    base_dir = Path(__file__).resolve().parent
    db_dir = (base_dir / "chroma_db").resolve()
    return base_dir, db_dir


def get_collection(db_dir: Path) -> Any:
    """连接已有的 parent-child Chroma collection。"""
    if not db_dir.exists():
        raise FileNotFoundError(f"ChromaDB directory does not exist: {db_dir}")

    if not db_dir.is_dir():
        raise NotADirectoryError(f"ChromaDB path is not a directory: {db_dir}")

    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_collection(name=COLLECTION_NAME)


def get_embedding_model() -> SentenceTransformer:
    """延迟初始化本地 SentenceTransformer 查询向量模型。"""
    global _embedding_model

    if _embedding_model is None:
        logger.info("Initializing SentenceTransformer model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    return _embedding_model


def encode_query(query: str) -> List[float]:
    """将查询文本编码为向量。"""
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


def tokenize_text(text: str) -> List[str]:
    """对文本进行轻量分词，供 BM25 使用。"""
    return re.findall(r"\w+", text.lower())


def normalize_page_value(page_value: Any) -> Optional[int]:
    """尽量将页码标准化为整数。"""
    if page_value is None:
        return None

    if isinstance(page_value, bool):
        return int(page_value)

    if isinstance(page_value, int):
        return page_value

    if isinstance(page_value, float):
        return int(page_value)

    page_text = str(page_value).strip()
    if not page_text:
        return None

    if page_text.isdigit():
        return int(page_text)

    return None


def build_page_display(page_value: Any) -> str:
    """将零基页码转换为一基展示页码。"""
    normalized_page = normalize_page_value(page_value)
    if normalized_page is None:
        return "N/A"
    return str(normalized_page + 1)


def fetch_parent_record(collection: Any, parent_node_id: str) -> Optional[Dict[str, Any]]:
    """根据父节点 ID 读取父节点全文与元数据。"""
    parent_result = collection.get(
        ids=[f"parent::{parent_node_id}"],
        include=["documents", "metadatas"],
    )

    parent_ids = parent_result.get("ids") or []
    if not parent_ids:
        logger.warning("Parent node not found for parent_node_id=%s", parent_node_id)
        return None

    parent_documents = parent_result.get("documents") or []
    parent_metadatas = parent_result.get("metadatas") or []

    document_text = parent_documents[0] if parent_documents else ""
    metadata = parent_metadatas[0] if parent_metadatas else {}
    page_value = metadata.get("page")

    return {
        "parent_node_id": parent_node_id,
        "source_file": metadata.get("source_file", "unknown"),
        "page": normalize_page_value(page_value),
        "page_display": build_page_display(page_value),
        "text": document_text,
    }


def collect_unique_parents_from_child_matches(
    collection: Any,
    child_metadatas: List[Optional[Dict[str, Any]]],
    child_scores: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    """根据子节点匹配结果提取唯一父节点，保留首次命中的排序。"""
    parent_results: List[Dict[str, Any]] = []
    seen_parent_ids = set()

    for rank_index, child_metadata in enumerate(child_metadatas, start=1):
        if not child_metadata:
            continue

        parent_node_id = child_metadata.get("parent_node_id")
        if not parent_node_id or parent_node_id in seen_parent_ids:
            continue

        parent_record = fetch_parent_record(collection=collection, parent_node_id=parent_node_id)
        if parent_record is None:
            continue

        parent_record["score"] = float(child_scores[rank_index - 1])
        parent_record["rank"] = rank_index
        parent_results.append(parent_record)
        seen_parent_ids.add(parent_node_id)

        if len(parent_results) >= top_k:
            break

    return parent_results


def count_nodes_by_type(collection: Any, node_type: str) -> int:
    """统计指定 node_type 的记录数量。"""
    result = collection.get(where={"node_type": node_type}, include=["metadatas"])
    return len(result.get("ids") or [])


def dense_retrieve(query: str, db_dir: Path, top_k: int = DEFAULT_DENSE_TOP_K) -> List[Dict[str, Any]]:
    """
    使用 Chroma 向量检索 child nodes，再回溯返回 parent nodes。

    返回值中的每条记录都对应唯一 parent node，而非 child node。
    """
    collection = get_collection(db_dir)
    child_count = count_nodes_by_type(collection, node_type="child")
    if child_count <= 0:
        return []

    candidate_k = min(max(top_k * 5, top_k), child_count)
    query_embedding = encode_query(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        where={"node_type": "child"},
        include=["metadatas", "distances"],
    )

    child_metadatas = (result.get("metadatas") or [[]])[0]
    child_distances = (result.get("distances") or [[]])[0]
    dense_scores = [1.0 / (1.0 + float(distance)) for distance in child_distances]

    return collect_unique_parents_from_child_matches(
        collection=collection,
        child_metadatas=child_metadatas,
        child_scores=dense_scores,
        top_k=top_k,
    )


def load_all_child_records(collection: Any) -> List[Dict[str, Any]]:
    """读取全部 child nodes，供 BM25 稀疏检索使用。"""
    result = collection.get(where={"node_type": "child"}, include=["documents", "metadatas"])
    child_ids = result.get("ids") or []
    child_documents = result.get("documents") or []
    child_metadatas = result.get("metadatas") or []

    records = []
    for child_id, document, metadata in zip(child_ids, child_documents, child_metadatas):
        if not document or not metadata:
            continue
        records.append(
            {
                "id": child_id,
                "text": document,
                "metadata": metadata,
            }
        )

    return records


def sparse_retrieve(query: str, db_dir: Path, top_k: int = DEFAULT_SPARSE_TOP_K) -> List[Dict[str, Any]]:
    """
    使用 BM25 检索 child nodes，再回溯返回 parent nodes。

    返回值中的每条记录都对应唯一 parent node，而非 child node。
    """
    collection = get_collection(db_dir)
    child_records = load_all_child_records(collection)
    if not child_records:
        return []

    tokenized_corpus = [tokenize_text(record["text"]) for record in child_records]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize_text(query)
    bm25_scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(child_records)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )

    candidate_k = min(max(top_k * 5, top_k), len(ranked_indexes))
    selected_indexes = ranked_indexes[:candidate_k]

    child_metadatas = [child_records[index]["metadata"] for index in selected_indexes]
    child_scores = [float(bm25_scores[index]) for index in selected_indexes]

    return collect_unique_parents_from_child_matches(
        collection=collection,
        child_metadatas=child_metadatas,
        child_scores=child_scores,
        top_k=top_k,
    )


def reciprocal_rank_fusion(result_lists: List[List[Dict[str, Any]]], k: int = RRF_K) -> List[Dict[str, Any]]:
    """对多个父节点结果列表执行 RRF 融合。"""
    fused_map: Dict[str, Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank_index, item in enumerate(result_list, start=1):
            parent_node_id = item["parent_node_id"]
            rrf_score = 1.0 / (k + rank_index)

            if parent_node_id not in fused_map:
                fused_map[parent_node_id] = {
                    "parent_node_id": parent_node_id,
                    "source_file": item["source_file"],
                    "page": item["page"],
                    "page_display": item["page_display"],
                    "text": item["text"],
                    "score": 0.0,
                }

            fused_map[parent_node_id]["score"] += rrf_score

    return sorted(fused_map.values(), key=lambda item: item["score"], reverse=True)


def hybrid_retrieve(query: str, db_dir: Path, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    执行 dense + sparse 混合检索，并用 RRF 融合为最终父节点结果。

    返回值严格使用应用层所需字段：label, source_file, page, page_display, score, text。
    """
    dense_results = dense_retrieve(query=query, db_dir=db_dir, top_k=max(top_k, DEFAULT_DENSE_TOP_K))
    sparse_results = sparse_retrieve(query=query, db_dir=db_dir, top_k=max(top_k, DEFAULT_SPARSE_TOP_K))
    fused_results = reciprocal_rank_fusion([dense_results, sparse_results], k=RRF_K)

    formatted_results = []
    for index, item in enumerate(fused_results[:top_k], start=1):
        formatted_results.append(
            {
                "label": f"E{index}",
                "source_file": item["source_file"],
                "page": item["page"],
                "page_display": item["page_display"],
                "score": round(float(item["score"]), 3),
                "text": item["text"],
            }
        )

    return formatted_results


def run_verification(query: str, db_dir: Path) -> Dict[str, Any]:
    """运行 dense / sparse / hybrid 检索并收集耗时与计数信息。"""
    collection = get_collection(db_dir)
    parent_count = count_nodes_by_type(collection, node_type="parent")
    child_count = count_nodes_by_type(collection, node_type="child")

    dense_start = time.perf_counter()
    dense_results = dense_retrieve(query=query, db_dir=db_dir, top_k=10)
    dense_elapsed = time.perf_counter() - dense_start

    sparse_start = time.perf_counter()
    sparse_results = sparse_retrieve(query=query, db_dir=db_dir, top_k=10)
    sparse_elapsed = time.perf_counter() - sparse_start

    hybrid_start = time.perf_counter()
    fused_results = hybrid_retrieve(query=query, db_dir=db_dir, top_k=3)
    hybrid_elapsed = time.perf_counter() - hybrid_start

    return {
        "parent_count": parent_count,
        "child_count": child_count,
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "fused_results": fused_results,
        "dense_elapsed": dense_elapsed,
        "sparse_elapsed": sparse_elapsed,
        "hybrid_elapsed": hybrid_elapsed,
    }


def print_fused_results(results: List[Dict[str, Any]]) -> None:
    """打印 top-k 融合结果，包含来源、页码与文本预览。"""
    separator = "-" * 100
    print(separator)
    print("Top Fused Results")
    print(separator)

    for result in results:
        preview = (result["text"] or "").replace("\n", " ").strip()
        if len(preview) > 240:
            preview = f"{preview[:240]}..."

        print(
            f"{result['label']} | source={result['source_file']} | "
            f"page={result['page_display']} | score={result['score']}"
        )
        print(preview)
        print(separator)


def print_summary(summary: Dict[str, Any]) -> None:
    """打印节点统计与各检索器耗时。"""
    print("Retrieval Summary")
    print(f"Total parent nodes : {summary['parent_count']}")
    print(f"Total child nodes  : {summary['child_count']}")
    print(f"Dense time         : {summary['dense_elapsed']:.4f}s")
    print(f"Sparse time        : {summary['sparse_elapsed']:.4f}s")
    print(f"Hybrid time        : {summary['hybrid_elapsed']:.4f}s")


def main() -> int:
    """执行脚本内置验证查询并打印结果摘要。"""
    configure_logging()

    try:
        base_dir, db_dir = resolve_paths()
        load_environment(base_dir)
        log_sqlite_runtime()

        logger.info("Project base directory: %s", base_dir)
        logger.info("Resolved ChromaDB directory: %s", db_dir)

        query = "What are the AML/CFT requirements for customer due diligence?"
        summary = run_verification(query=query, db_dir=db_dir)
        print_fused_results(summary["fused_results"])
        print_summary(summary)
        return 0
    except FileNotFoundError as exc:
        logger.error("Missing required path: %s", exc)
        return 1
    except NotADirectoryError as exc:
        logger.error("Invalid directory configuration: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure while running hybrid retrieval: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
