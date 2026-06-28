# ARCHIVED — 2024 实验版本，当前入口为 app.py

import sys

# -----------------------------
# SQLite3 兼容性猴子补丁
# -----------------------------
# 必须放在文件最顶部，并且早于 chromadb / langchain 相关导入。
# 目的：在本地系统 sqlite3 版本过低时，优先尝试切换到 pysqlite3。
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# 全局配置常量
# -----------------------------
COLLECTION_NAME = "regtech_pdf_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SQLITE_VERSION = (3, 35, 0)

logger = logging.getLogger(__name__)


def _parse_sqlite_version(version: str) -> Tuple[int, int, int]:
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
    """配置标准日志，方便在本地排查数据库连接与校验问题。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )



def load_environment(base_dir: Path) -> None:
    """从项目根目录加载 .env 文件。"""
    env_path = base_dir / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info("Environment variables loaded from: %s", env_path)



def resolve_paths() -> Tuple[Path, Path]:
    """
    统一解析项目根目录与本地 ChromaDB 持久化目录。

    这里使用 pathlib.resolve() 生成绝对路径，确保在不同 shell 启动目录下
    都能稳定定位到同一个数据库目录。
    """
    base_dir = Path(__file__).resolve().parent
    db_dir = (base_dir / "chroma_db").resolve()
    return base_dir, db_dir



def log_sqlite_runtime() -> None:
    """记录当前 SQLite 后端与版本，便于定位环境兼容性问题。"""
    logger.info("Active sqlite backend: %s", ACTIVE_SQLITE_BACKEND)
    logger.info("Active sqlite version: %s", ACTIVE_SQLITE_VERSION)



def create_embeddings() -> HuggingFaceEmbeddings:
    """
    初始化本地 HuggingFace 向量模型。

    说明：验证脚本虽然主要做读操作，但仍显式初始化与主构建脚本一致的
    embedding 配置，以确认检索侧和写入侧的模型约定保持一致。
    """
    logger.info("Initializing local HuggingFace embeddings model: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )



def connect_to_chroma(db_dir: Path) -> Tuple[chromadb.PersistentClient, Any]:
    """
    连接本地 ChromaDB 并校验目标 collection 是否存在。

    返回值中第二个对象为底层 collection 句柄，后续直接用它做 count/get，
    可以更精确地控制抽样读取行为。
    """
    if not db_dir.exists():
        raise FileNotFoundError(f"ChromaDB directory does not exist: {db_dir}")

    if not db_dir.is_dir():
        raise NotADirectoryError(f"ChromaDB path is not a directory: {db_dir}")

    client = chromadb.PersistentClient(path=str(db_dir))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open collection '{COLLECTION_NAME}' from database directory: {db_dir}"
        ) from exc

    # 这里显式初始化 LangChain 的 Chroma 包装器，验证读侧配置与主脚本兼容。
    embeddings = create_embeddings()
    Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )

    return client, collection



def fetch_sample_chunk(collection: Any) -> Tuple[str, Dict[str, Any]]:
    """
    获取恰好一个样本文档块及其元数据。

    若集合为空，则立即抛出异常，使 QA 验证脚本以失败状态退出，便于接入
    CI/CD 或本地发布前检查流程。
    """
    total_chunks = collection.count()
    if total_chunks <= 0:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' is empty. Validation failed because no chunks are available."
        )

    sample = collection.get(limit=1, include=["documents", "metadatas"])
    documents = sample.get("documents") or []
    metadatas = sample.get("metadatas") or []

    if not documents:
        raise RuntimeError("Collection returned no documents when attempting to fetch a sample chunk.")

    sample_text = documents[0] or ""
    sample_metadata: Optional[Dict[str, Any]] = None
    if metadatas:
        sample_metadata = metadatas[0]

    return sample_text, sample_metadata or {}



def print_verification_report(total_chunks: int, sample_text: str, sample_metadata: Dict[str, Any]) -> None:
    """以清晰、可人工审阅的版式打印数据库验证结果。"""
    separator = "-" * 100
    print(separator)
    print("ChromaDB Verification Report")
    print(separator)
    print(f"Collection Name : {COLLECTION_NAME}")
    print(f"Total Chunks    : {total_chunks}")
    print(separator)
    print("Sample Metadata")
    print(separator)
    print(f"source_file     : {sample_metadata.get('source_file', 'N/A')}")
    print(f"page            : {sample_metadata.get('page', 'N/A')}")
    print(separator)
    print("Sample Chunk Text")
    print(separator)
    print(sample_text)
    print(separator)



def main() -> int:
    """
    执行本地 ChromaDB 验证流程。

    核心步骤：
    1. 加载环境变量与路径。
    2. 校验 SQLite/Chroma 运行环境。
    3. 连接本地数据库与目标 collection。
    4. 输出总 chunk 数量。
    5. 抽取一个样本 chunk，打印正文与元数据。
    """
    configure_logging()

    try:
        base_dir, db_dir = resolve_paths()
        load_environment(base_dir)
        log_sqlite_runtime()

        logger.info("Project base directory: %s", base_dir)
        logger.info("Resolved ChromaDB directory: %s", db_dir)

        _, collection = connect_to_chroma(db_dir)

        total_chunks = collection.count()
        logger.info("Collection '%s' contains %s chunk(s).", COLLECTION_NAME, total_chunks)

        sample_text, sample_metadata = fetch_sample_chunk(collection)
        print_verification_report(total_chunks, sample_text, sample_metadata)
        return 0
    except FileNotFoundError as exc:
        logger.error("Missing required path: %s", exc)
        return 1
    except NotADirectoryError as exc:
        logger.error("Invalid directory configuration: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Database verification failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure while verifying ChromaDB: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
