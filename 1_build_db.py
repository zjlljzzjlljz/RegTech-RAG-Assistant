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

import sqlite3


# 在导入 Chroma 之前先做 SQLite 运行时校验，避免报错位置过深、不利于排障。
MIN_SQLITE_VERSION = (3, 35, 0)


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

import logging
from pathlib import Path
from typing import List, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# 全局配置常量
# -----------------------------
COLLECTION_NAME = "regtech_pdf_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

logger = logging.getLogger(__name__)


def log_sqlite_runtime() -> None:
    """记录当前 SQLite 后端与版本，便于排查 Chroma 运行环境问题。"""
    logger.info("Active sqlite backend: %s", ACTIVE_SQLITE_BACKEND)
    logger.info("Active sqlite version: %s", ACTIVE_SQLITE_VERSION)


def configure_logging() -> None:
    """配置标准日志，输出时间戳、等级和消息内容。"""
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


def resolve_paths() -> Tuple[Path, Path, Path]:
    """
    统一解析脚本所在项目目录、数据目录和 ChromaDB 持久化目录。

    使用 pathlib.resolve() 保证路径为绝对路径，避免在不同操作系统
    或不同启动目录下出现相对路径歧义。
    """
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


def load_pdf_documents(pdf_files: List[Path]) -> List[Document]:
    """
    加载所有 PDF 页面，并显式保留 source_file 与 page 元数据。

    设计说明：
    1. 每个 PDF 单独 try-except，保证单个坏文件不会中断整个批处理。
    2. 每个页面被标准化为 LangChain Document，避免后续切块阶段依赖
       加载器内部的元数据字段结构。
    3. 仅保留当前检索链路真正需要的字段：source_file、page。
    """
    documents: List[Document] = []

    for pdf_path in pdf_files:
        try:
            logger.info("Loading PDF: %s", pdf_path)
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()

            if not pages:
                logger.warning("No pages were loaded from PDF: %s", pdf_path.name)
                continue

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
                normalized_doc = Document(
                    page_content=page_content,
                    metadata={
                        "source_file": pdf_path.name,
                        "page": page_number,
                    },
                )
                documents.append(normalized_doc)

            logger.info(
                "Loaded %s non-empty pages from PDF: %s",
                len(pages),
                pdf_path.name,
            )
        except FileNotFoundError:
            logger.exception("PDF file not found: %s", pdf_path)
        except Exception as exc:
            # 这里统一捕获损坏 PDF、解析失败等异常，继续处理其他文件。
            logger.exception("Failed to load PDF '%s': %s", pdf_path.name, exc)

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    对文档进行语义友好的递归切块。

    这里使用较大的 chunk_size 和适中的 overlap，目的是尽量保留
    监管条款、定义段落、长句法之间的上下文连续性，降低召回时
    断章取义的概率。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)
    cleaned_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    return cleaned_chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    """初始化本地 CPU 向量模型，确保敏感文档不离开本地环境。"""
    logger.info("Initializing local HuggingFace embeddings model: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def reset_collection(client: chromadb.PersistentClient, collection_name: str) -> None:
    """
    仅删除目标 collection，实现幂等重建。

    注意：这里不会删除整个 chroma_db 目录，因此其他 collection 或底层
    持久化结构不会被误删。
    """
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


def build_vector_store(chunks: List[Document], db_dir: Path) -> int:
    """
    构建并持久化 Chroma 向量库，返回最终成功写入的 chunk 数量。

    流程说明：
    1. 使用 PersistentClient 指向本地目录。
    2. 若目标 collection 已存在，则先删除再重建，确保幂等。
    3. 通过 LangChain 的 Chroma 封装写入文档和 embedding。
    4. 最后读取底层 collection.count() 做精确核验。
    """
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    reset_collection(client=client, collection_name=COLLECTION_NAME)

    embeddings = create_embeddings()
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )

    chunk_ids = [
        f"{chunk.metadata.get('source_file', 'unknown')}::page-{chunk.metadata.get('page', 'na')}::chunk-{index}"
        for index, chunk in enumerate(chunks)
    ]

    logger.info("Writing %s chunks into ChromaDB collection '%s'", len(chunks), COLLECTION_NAME)
    vector_store.add_documents(documents=chunks, ids=chunk_ids)

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    stored_count = client.get_collection(name=COLLECTION_NAME).count()
    return stored_count


def main() -> int:
    """执行本地 PDF -> ChromaDB 的完整构建流程。"""
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
        chunks = chunk_documents(documents)
        if not chunks:
            logger.error("Document chunking produced zero chunks. Aborting ingestion.")
            return 1

        logger.info("Generated %s chunk(s) after semantic splitting.", len(chunks))
        stored_count = build_vector_store(chunks, db_dir)

        logger.info(
            "Successfully stored %s chunks in ChromaDB collection '%s'.",
            stored_count,
            COLLECTION_NAME,
        )
        return 0
    except FileNotFoundError as exc:
        logger.error("Missing required path: %s", exc)
        return 1
    except NotADirectoryError as exc:
        logger.error("Invalid directory configuration: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected failure while building ChromaDB: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
