from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

MIN_SQLITE_VERSION = (3, 35, 0)
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import sqlite3  # noqa: E402


@dataclass(frozen=True)
class PathSettings:
    project_root: Path
    data_dir: Path
    raw_pdf_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class MilvusSettings:
    host: str
    port: int
    user: str
    password: str
    database: str
    alias: str
    collection_name: str
    consistency_level: str
    dense_index_type: str
    sparse_index_type: str
    dense_metric_type: str
    sparse_metric_type: str
    search_probe: int

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class LLMSettings:
    model: str
    max_tokens: int
    effort: str
    enable_adaptive_thinking: bool
    thinking_display: str
    enable_context_compaction: bool
    compaction_beta: str
    prompt_cache_ttl: str


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    rrf_top_k: int
    rerank_top_k: int
    rrf_k: int
    rerank_score_threshold: float
    max_audit_iterations: int
    async_workers: int
    max_history_turns: int
    compression_trigger_chars: int
    compression_keep_recent_turns: int


@dataclass(frozen=True)
class InferenceSettings:
    embedding_model_name: str
    reranker_model_name: str
    embedding_service_url: str | None
    reranker_service_url: str | None
    request_timeout_seconds: int
    prefer_local_fallback: bool


@dataclass(frozen=True)
class AppSettings:
    log_level: str
    page_title: str
    page_icon: str
    streamlit_port: int


@dataclass(frozen=True)
class Settings:
    paths: PathSettings
    milvus: MilvusSettings
    llm: LLMSettings
    retrieval: RetrievalSettings
    inference: InferenceSettings
    app: AppSettings
    anthropic_api_key: str | None
    anthropic_base_url: str | None
    anthropic_auth_token: str | None


class ConfigurationError(RuntimeError):
    """Raised when runtime configuration is incomplete or invalid."""



def _parse_sqlite_version(version: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for part in version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break

    while len(parts) < 3:
        parts.append(0)

    return tuple(parts[:3])



def ensure_sqlite_compatibility() -> None:
    version_tuple = _parse_sqlite_version(sqlite3.sqlite_version)
    if version_tuple < MIN_SQLITE_VERSION:
        raise ConfigurationError(
            "Unsupported sqlite3 runtime detected. "
            f"Active backend: {getattr(sqlite3, '__file__', sqlite3.__name__)}; "
            f"version: {sqlite3.sqlite_version}. "
            "Install pysqlite3-binary or run Python linked against SQLite >= 3.35.0."
        )



def resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent



def load_environment() -> Path:
    project_root = resolve_project_root()
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    return env_path



def configure_logging(level: str | None = None) -> None:
    resolved_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )



def _resolve_paths(project_root: Path) -> PathSettings:
    data_dir = project_root / "data"
    raw_pdf_dir = data_dir / "raw_pdfs"
    report_dir = project_root / "reports"
    return PathSettings(
        project_root=project_root,
        data_dir=data_dir,
        raw_pdf_dir=raw_pdf_dir,
        report_dir=report_dir,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    ensure_sqlite_compatibility()
    load_environment()
    project_root = resolve_project_root()
    paths = _resolve_paths(project_root)

    return Settings(
        paths=paths,
        milvus=MilvusSettings(
            host=os.getenv("MILVUS_HOST", "127.0.0.1"),
            port=int(os.getenv("MILVUS_PORT", "19530")),
            user=os.getenv("MILVUS_USER", ""),
            password=os.getenv("MILVUS_PASSWORD", ""),
            database=os.getenv("MILVUS_DATABASE", "default"),
            alias=os.getenv("MILVUS_ALIAS", "default"),
            collection_name=os.getenv("MILVUS_COLLECTION", "regtech_compliance_chunks"),
            consistency_level=os.getenv("MILVUS_CONSISTENCY_LEVEL", "Session"),
            dense_index_type=os.getenv("MILVUS_DENSE_INDEX_TYPE", "HNSW"),
            sparse_index_type=os.getenv("MILVUS_SPARSE_INDEX_TYPE", "SPARSE_INVERTED_INDEX"),
            dense_metric_type=os.getenv("MILVUS_DENSE_METRIC", "COSINE"),
            sparse_metric_type=os.getenv("MILVUS_SPARSE_METRIC", "IP"),
            search_probe=int(os.getenv("MILVUS_SEARCH_PROBE", "64")),
        ),
        llm=LLMSettings(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4000")),
            effort=os.getenv("ANTHROPIC_EFFORT", "high"),
            enable_adaptive_thinking=os.getenv("ANTHROPIC_ENABLE_THINKING", "true").lower() == "true",
            thinking_display=os.getenv("ANTHROPIC_THINKING_DISPLAY", "summarized"),
            enable_context_compaction=os.getenv("ANTHROPIC_ENABLE_COMPACTION", "true").lower() == "true",
            compaction_beta=os.getenv("ANTHROPIC_COMPACTION_BETA", "compact-2026-01-12"),
            prompt_cache_ttl=os.getenv("ANTHROPIC_PROMPT_CACHE_TTL", "1h"),
        ),
        retrieval=RetrievalSettings(
            dense_top_k=int(os.getenv("RETRIEVAL_DENSE_TOP_K", "50")),
            sparse_top_k=int(os.getenv("RETRIEVAL_SPARSE_TOP_K", "50")),
            rrf_top_k=int(os.getenv("RETRIEVAL_RRF_TOP_K", "20")),
            rerank_top_k=int(os.getenv("RETRIEVAL_RERANK_TOP_K", "8")),
            rrf_k=int(os.getenv("RETRIEVAL_RRF_K", "60")),
            rerank_score_threshold=float(os.getenv("RETRIEVAL_RERANK_SCORE_THRESHOLD", "0.25")),
            max_audit_iterations=int(os.getenv("MAX_AUDIT_ITERATIONS", "3")),
            async_workers=int(os.getenv("ASYNC_RETRIEVAL_WORKERS", "4")),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "12")),
            compression_trigger_chars=int(os.getenv("HISTORY_COMPRESSION_TRIGGER_CHARS", "12000")),
            compression_keep_recent_turns=int(os.getenv("HISTORY_KEEP_RECENT_TURNS", "4")),
        ),
        inference=InferenceSettings(
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            reranker_model_name=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large"),
            embedding_service_url=(os.getenv("EMBEDDING_SERVICE_URL") or "").strip() or None,
            reranker_service_url=(os.getenv("RERANKER_SERVICE_URL") or "").strip() or None,
            request_timeout_seconds=int(os.getenv("INFERENCE_REQUEST_TIMEOUT_SECONDS", "60")),
            prefer_local_fallback=os.getenv("PREFER_LOCAL_INFERENCE_FALLBACK", "true").lower() == "true",
        ),
        app=AppSettings(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            page_title=os.getenv("STREAMLIT_PAGE_TITLE", "RegTech 3.0 Compliance Assistant"),
            page_icon=os.getenv("STREAMLIT_PAGE_ICON", "🛡️"),
            streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
        ),
        anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip() or None,
        anthropic_base_url=(os.getenv("ANTHROPIC_BASE_URL") or "").strip() or None,
        anthropic_auth_token=(os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip() or None,
    )


def get_anthropic_client(settings: Settings | None = None) -> "Anthropic":
    """Create an Anthropic client with header-stripping transport for proxy compatibility."""
    import httpx
    from anthropic import Anthropic

    class _CleanTransport(httpx.BaseTransport):
        _BLOCKED = {"x-stainless", "accept", "user-agent", "accept-encoding", "connection", "cf-"}

        def __init__(self) -> None:
            self._t = httpx.HTTPTransport()

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            for h in list(request.headers.keys()):
                if any(h.startswith(p) or h == p for p in self._BLOCKED):
                    del request.headers[h]
            return self._t.handle_request(request)

    resolved = settings or get_settings()
    return Anthropic(
        api_key=resolved.anthropic_api_key,
        base_url=resolved.anthropic_base_url or None,
        http_client=httpx.Client(transport=_CleanTransport(), timeout=resolved.inference.request_timeout_seconds),
    )


__all__ = [
    "AppSettings",
    "ConfigurationError",
    "InferenceSettings",
    "LLMSettings",
    "MilvusSettings",
    "PathSettings",
    "RetrievalSettings",
    "Settings",
    "configure_logging",
    "ensure_sqlite_compatibility",
    "get_anthropic_client",
    "get_settings",
    "load_environment",
    "resolve_project_root",
]
