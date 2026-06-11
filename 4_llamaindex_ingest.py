import json
import sys

# -----------------------------
# SQLite3 兼容性猴子补丁
# -----------------------------
# 必须放在文件最顶部，并且早于 chromadb / llama_index 相关导入。
# 目的：在本地系统 sqlite3 版本过低时，优先尝试切换到 pysqlite3。
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import MetadataMode, NodeRelationship
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MIN_SQLITE_VERSION = (3, 35, 0)
COLLECTION_NAME = "regtech_parent_child_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PARENT_CHUNK_SIZE = 1024
CHILD_CHUNK_SIZE = 256
EMBED_BATCH_SIZE = 64

logger = logging.getLogger(__name__)


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


def resolve_paths() -> Tuple[Path, Path, Path]:
    """统一解析项目目录、数据目录和 ChromaDB 持久化目录。"""
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / "data").resolve()
    db_dir = (base_dir / "chroma_db").resolve()
    return base_dir, data_dir, db_dir


def discover_pdf_files(data_dir: Path) -> List[Path]:
    """扫描数据目录中的所有 PDF 文件。"""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    pdf_files = sorted(data_dir.glob("*.pdf"))
    return [pdf_file.resolve() for pdf_file in pdf_files if pdf_file.is_file()]


def load_pdf_documents(pdf_files: List[Path]) -> List[LlamaDocument]:
    """
    将 PDF 按页加载后转换为 LlamaIndex Document。

    保留 source_file 与 page 元数据，保证后续父子节点切块后仍能追溯来源。
    """
    documents: List[LlamaDocument] = []

    for pdf_path in pdf_files:
        try:
            logger.info("Loading PDF: %s", pdf_path)
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()

            if not pages:
                logger.warning("No pages were loaded from PDF: %s", pdf_path.name)
                continue

            loaded_pages = 0
            for page_doc in pages:
                page_content = page_doc.page_content.strip()
                if not page_content:
                    logger.warning(
                        "Skipping empty page in PDF '%s' (page=%s)",
                        pdf_path.name,
                        page_doc.metadata.get("page"),
                    )
                    continue

                page_number = page_doc.metadata.get("page")
                documents.append(
                    LlamaDocument(
                        text=page_content,
                        metadata={
                            "source_file": pdf_path.name,
                            "page": page_number,
                        },
                    )
                )
                loaded_pages += 1

            logger.info("Loaded %s non-empty pages from PDF: %s", loaded_pages, pdf_path.name)
        except FileNotFoundError:
            logger.exception("PDF file not found: %s", pdf_path)
        except Exception as exc:
            logger.exception("Failed to load PDF '%s': %s", pdf_path.name, exc)

    return documents


def build_hierarchical_nodes(documents: List[LlamaDocument]) -> Tuple[List[Any], List[Any]]:
    """使用 HierarchicalNodeParser 构建父子节点。"""
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE],
    )

    all_nodes = parser.get_nodes_from_documents(documents)
    child_nodes = list(get_leaf_nodes(all_nodes))
    child_node_ids = {node.node_id for node in child_nodes}
    parent_nodes = [node for node in all_nodes if node.node_id not in child_node_ids]
    return parent_nodes, child_nodes


def create_embeddings() -> HuggingFaceEmbedding:
    """初始化本地 CPU 向量模型，确保敏感文档不离开本地环境。"""
    logger.info("Initializing local HuggingFace embeddings model: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        device="cpu",
        normalize=True,
    )


def reset_collection(client: chromadb.PersistentClient, collection_name: str) -> None:
    """仅删除目标 collection，实现幂等重建。"""
    existing_collections = client.list_collections()
    existing_names = {
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in existing_collections
    }

    if collection_name in existing_names:
        logger.info("Existing collection '%s' found. Deleting it before rebuild.", collection_name)
        client.delete_collection(name=collection_name)
    else:
        logger.info("Collection '%s' does not exist yet. Creating a new one.", collection_name)


def extract_parent_node_id(node: Any) -> Optional[str]:
    """从节点关系中提取父节点 ID。"""
    relationships = getattr(node, "relationships", {}) or {}
    parent_relationship = relationships.get(NodeRelationship.PARENT)
    if parent_relationship is None:
        return None

    if isinstance(parent_relationship, list):
        for related in parent_relationship:
            parent_id = getattr(related, "node_id", None)
            if parent_id:
                return parent_id
        return None

    return getattr(parent_relationship, "node_id", None)


def build_parent_child_map(child_nodes: List[Any]) -> Dict[str, List[str]]:
    """基于子节点关系构建 parent -> children 映射。"""
    parent_child_map: Dict[str, List[str]] = {}

    for child_node in child_nodes:
        parent_node_id = extract_parent_node_id(child_node)
        if not parent_node_id:
            logger.warning("Child node '%s' does not contain a parent relationship.", child_node.node_id)
            continue
        parent_child_map.setdefault(parent_node_id, []).append(child_node.node_id)

    return parent_child_map


def _coerce_page_value(page_value: Any) -> Optional[Any]:
    """将页码标准化为 Chroma 兼容的简单标量类型。"""
    if page_value is None:
        return None

    if isinstance(page_value, (int, float, bool)):
        return page_value

    page_text = str(page_value).strip()
    if not page_text:
        return None

    if page_text.isdigit():
        return int(page_text)

    return page_text


def _get_node_text(node: Any) -> str:
    """提取节点文本内容，并移除空白。"""
    return node.get_content(metadata_mode=MetadataMode.NONE).strip()


def build_node_records(parent_nodes: List[Any], child_nodes: List[Any]) -> List[Dict[str, Any]]:
    """将父子节点转换为可写入 Chroma 的记录结构。"""
    parent_child_map = build_parent_child_map(child_nodes)
    records: List[Dict[str, Any]] = []

    for parent_node in parent_nodes:
        parent_text = _get_node_text(parent_node)
        if not parent_text:
            continue

        metadata = getattr(parent_node, "metadata", {}) or {}
        record_metadata: Dict[str, Any] = {
            "node_id": parent_node.node_id,
            "node_type": "parent",
            "source_file": str(metadata.get("source_file", "unknown")),
            "char_length": len(parent_text),
            "child_node_ids": json.dumps(parent_child_map.get(parent_node.node_id, []), ensure_ascii=False),
        }
        page_value = _coerce_page_value(metadata.get("page"))
        if page_value is not None:
            record_metadata["page"] = page_value

        records.append(
            {
                "id": f"parent::{parent_node.node_id}",
                "document": parent_text,
                "metadata": record_metadata,
            }
        )

    for child_node in child_nodes:
        child_text = _get_node_text(child_node)
        if not child_text:
            continue

        metadata = getattr(child_node, "metadata", {}) or {}
        record_metadata = {
            "node_id": child_node.node_id,
            "node_type": "child",
            "source_file": str(metadata.get("source_file", "unknown")),
            "char_length": len(child_text),
        }

        page_value = _coerce_page_value(metadata.get("page"))
        if page_value is not None:
            record_metadata["page"] = page_value

        parent_node_id = extract_parent_node_id(child_node)
        if parent_node_id:
            record_metadata["parent_node_id"] = parent_node_id

        records.append(
            {
                "id": f"child::{child_node.node_id}",
                "document": child_text,
                "metadata": record_metadata,
            }
        )

    return records


def write_records_to_collection(
    collection: Any,
    records: List[Dict[str, Any]],
    embed_model: HuggingFaceEmbedding,
) -> int:
    """按批次计算 embedding 并写入 Chroma collection。"""
    for start_index in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[start_index : start_index + EMBED_BATCH_SIZE]
        batch_documents = [record["document"] for record in batch]
        batch_ids = [record["id"] for record in batch]
        batch_metadatas = [record["metadata"] for record in batch]
        batch_embeddings = embed_model.get_text_embedding_batch(batch_documents)

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings,
        )

    return collection.count()


def build_vector_store(parent_nodes: List[Any], child_nodes: List[Any], db_dir: Path) -> int:
    """构建并持久化父子节点 Chroma collection。"""
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    reset_collection(client=client, collection_name=COLLECTION_NAME)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embed_model = create_embeddings()
    records = build_node_records(parent_nodes=parent_nodes, child_nodes=child_nodes)
    if not records:
        raise RuntimeError("Hierarchical node conversion produced zero writable records.")

    logger.info(
        "Writing %s hierarchical nodes into ChromaDB collection '%s'.",
        len(records),
        COLLECTION_NAME,
    )
    return write_records_to_collection(collection=collection, records=records, embed_model=embed_model)


def main() -> int:
    """执行本地 PDF -> LlamaIndex 父子节点 -> ChromaDB 的完整构建流程。"""
    configure_logging()

    try:
        base_dir, data_dir, db_dir = resolve_paths()
        load_environment(base_dir)
        log_sqlite_runtime()

        logger.info("Project base directory: %s", base_dir)
        logger.info("Resolved data directory: %s", data_dir)
        logger.info("Resolved ChromaDB directory: %s", db_dir)

        pdf_files = discover_pdf_files(data_dir)
        if not pdf_files:
            logger.warning("No PDF files found in data directory: %s", data_dir)
            return 0

        logger.info("Discovered %s PDF file(s) for ingestion.", len(pdf_files))
        documents = load_pdf_documents(pdf_files)
        if not documents:
            logger.error("No valid document pages were loaded. Aborting ingestion.")
            return 1

        logger.info("Loaded %s valid document page(s).", len(documents))
        parent_nodes, child_nodes = build_hierarchical_nodes(documents)
        if not parent_nodes:
            logger.error("Hierarchical chunking produced zero parent nodes. Aborting ingestion.")
            return 1

        if not child_nodes:
            logger.error("Hierarchical chunking produced zero child nodes. Aborting ingestion.")
            return 1

        logger.info("Generated %s parent node(s).", len(parent_nodes))
        logger.info("Generated %s child node(s).", len(child_nodes))

        stored_count = build_vector_store(parent_nodes=parent_nodes, child_nodes=child_nodes, db_dir=db_dir)
        logger.info(
            "Successfully stored %s hierarchical nodes in ChromaDB collection '%s'.",
            stored_count,
            COLLECTION_NAME,
        )

        print(f"Parent nodes: {len(parent_nodes)}")
        print(f"Child nodes: {len(child_nodes)}")
        print(f"Stored records: {stored_count}")
        return 0
    except FileNotFoundError as exc:
        logger.error("Missing required path: %s", exc)
        return 1
    except NotADirectoryError as exc:
        logger.error("Invalid directory configuration: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure while building hierarchical ChromaDB: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
