from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

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
class LLMRoleSettings:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    max_tokens: int
    temperature: float
    timeout_seconds: int


@dataclass(frozen=True)
class LLMRolesSettings:
    planner: LLMRoleSettings
    draft: LLMRoleSettings
    auditor: LLMRoleSettings
    judge: LLMRoleSettings


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
    hyde_score_threshold: float
    hyde_margin_threshold: float
    planner_cache_ttl_seconds: int
    parent_backfill: bool
    original_query_rrf_weight: float
    dense_rrf_weight: float
    sparse_rrf_weight: float


@dataclass(frozen=True)
class ChunkingSettings:
    parent_tokens: int
    child_tokens: int
    overlap_tokens: int
    version: str


@dataclass(frozen=True)
class StorageSettings:
    database_url: str | None
    sqlite_path: str | None


@dataclass(frozen=True)
class InferenceSettings:
    embedding_model_name: str
    reranker_model_name: str
    reranker_enabled: bool
    embedding_service_url: str | None
    reranker_service_url: str | None
    request_timeout_seconds: int
    prefer_local_fallback: bool
    nli_model_name: str
    nli_service_url: str | None
    nli_entailment_threshold: float
    nli_enabled: bool
    embedding_batch_size: int
    embedding_max_length: int


@dataclass(frozen=True)
class AppSettings:
    log_level: str
    page_title: str
    page_icon: str
    streamlit_port: int


@dataclass(frozen=True)
class DeploymentSettings:
    mode: str


@dataclass(frozen=True)
class Settings:
    deployment: DeploymentSettings
    paths: PathSettings
    milvus: MilvusSettings
    llm: LLMSettings
    llm_roles: LLMRolesSettings
    retrieval: RetrievalSettings
    chunking: ChunkingSettings
    inference: InferenceSettings
    storage: StorageSettings
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



def load_environment(
    env_path: Path | None = None, mode: str | None = None
) -> Path:
    resolved_path = env_path or resolve_project_root() / ".env"
    resolved_mode = (mode or os.getenv("DEPLOYMENT_MODE", "host")).strip().lower()
    if resolved_mode == "host":
        load_dotenv(dotenv_path=resolved_path, override=False)
    return resolved_path



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
    explicit_deployment_mode = os.getenv("DEPLOYMENT_MODE")
    load_environment(mode=explicit_deployment_mode or "host")
    deployment_mode = os.getenv("DEPLOYMENT_MODE", "host").strip().lower()
    if deployment_mode not in {"host", "api-cpu", "gpu-self-hosted"}:
        raise ConfigurationError(
            "DEPLOYMENT_MODE must be one of: host, api-cpu, gpu-self-hosted"
        )
    if explicit_deployment_mode is None and deployment_mode != "host":
        raise ConfigurationError(
            "container deployment modes must be set explicitly in the process "
            "environment or Compose, not in the project .env"
        )
    project_root = resolve_project_root()
    paths = _resolve_paths(project_root)

    legacy_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    legacy_base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip() or None
    legacy_api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip() or None

    def _role(prefix: str, default_model: str, default_max_tokens: int) -> LLMRoleSettings:
        provider = os.getenv(f"{prefix}_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "anthropic")).lower()
        provider_default_model = legacy_model if provider == "anthropic" else default_model
        return LLMRoleSettings(
            provider=provider,
            model=os.getenv(f"{prefix}_LLM_MODEL", provider_default_model),
            base_url=(os.getenv(f"{prefix}_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or legacy_base_url or "").strip() or None,
            api_key=(
                os.environ[f"{prefix}_LLM_API_KEY"]
                if f"{prefix}_LLM_API_KEY" in os.environ
                else os.getenv("LLM_API_KEY") or legacy_api_key or ""
            ).strip()
            or None,
            max_tokens=int(os.getenv(f"{prefix}_LLM_MAX_TOKENS", str(default_max_tokens))),
            temperature=float(os.getenv(f"{prefix}_LLM_TEMPERATURE", "0")),
            timeout_seconds=int(os.getenv(f"{prefix}_LLM_TIMEOUT_SECONDS", "120")),
        )

    settings = Settings(
        deployment=DeploymentSettings(mode=deployment_mode),
        paths=paths,
        milvus=MilvusSettings(
            host=os.getenv("MILVUS_HOST", "127.0.0.1"),
            port=int(os.getenv("MILVUS_PORT", "19530")),
            user=os.getenv("MILVUS_USER", ""),
            password=os.getenv("MILVUS_PASSWORD", ""),
            database=os.getenv("MILVUS_DATABASE", "default"),
            alias=os.getenv("MILVUS_ALIAS", "default"),
            collection_name=os.getenv("MILVUS_COLLECTION", "regtech_compliance_chunks_v2"),
            consistency_level=os.getenv("MILVUS_CONSISTENCY_LEVEL", "Session"),
            dense_index_type=os.getenv("MILVUS_DENSE_INDEX_TYPE", "HNSW"),
            sparse_index_type=os.getenv("MILVUS_SPARSE_INDEX_TYPE", "SPARSE_INVERTED_INDEX"),
            dense_metric_type=os.getenv("MILVUS_DENSE_METRIC", "COSINE"),
            sparse_metric_type=os.getenv("MILVUS_SPARSE_METRIC", "IP"),
            search_probe=int(os.getenv("MILVUS_SEARCH_PROBE", "64")),
        ),
        llm=LLMSettings(
            model=legacy_model,
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4000")),
            effort=os.getenv("ANTHROPIC_EFFORT", "high"),
            enable_adaptive_thinking=os.getenv("ANTHROPIC_ENABLE_THINKING", "true").lower() == "true",
            thinking_display=os.getenv("ANTHROPIC_THINKING_DISPLAY", "summarized"),
            enable_context_compaction=os.getenv("ANTHROPIC_ENABLE_COMPACTION", "true").lower() == "true",
            compaction_beta=os.getenv("ANTHROPIC_COMPACTION_BETA", "compact-2026-01-12"),
            prompt_cache_ttl=os.getenv("ANTHROPIC_PROMPT_CACHE_TTL", "1h"),
        ),
        llm_roles=LLMRolesSettings(
            planner=_role("PLANNER", "Qwen/Qwen2.5-7B-Instruct", 1024),
            draft=_role("DRAFT", "Qwen/Qwen2.5-72B-Instruct-AWQ", 4000),
            auditor=_role("AUDITOR", "Qwen/Qwen2.5-72B-Instruct-AWQ", 2000),
            judge=_role("JUDGE", "Qwen/Qwen2.5-72B-Instruct-AWQ", 2000),
        ),
        retrieval=RetrievalSettings(
            dense_top_k=int(os.getenv("RETRIEVAL_DENSE_TOP_K", "50")),
            sparse_top_k=int(os.getenv("RETRIEVAL_SPARSE_TOP_K", "50")),
            rrf_top_k=int(os.getenv("RETRIEVAL_RRF_TOP_K", "20")),
            rerank_top_k=int(os.getenv("RETRIEVAL_RERANK_TOP_K", "8")),
            rrf_k=int(os.getenv("RETRIEVAL_RRF_K", "60")),
            rerank_score_threshold=float(os.getenv("RETRIEVAL_RERANK_SCORE_THRESHOLD", "0.0")),
            max_audit_iterations=int(os.getenv("MAX_AUDIT_ITERATIONS", "3")),
            async_workers=int(os.getenv("ASYNC_RETRIEVAL_WORKERS", "4")),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "12")),
            compression_trigger_chars=int(os.getenv("HISTORY_COMPRESSION_TRIGGER_CHARS", "12000")),
            compression_keep_recent_turns=int(os.getenv("HISTORY_KEEP_RECENT_TURNS", "4")),
            hyde_score_threshold=float(os.getenv("HYDE_SCORE_THRESHOLD", "0.25")),
            hyde_margin_threshold=float(os.getenv("HYDE_MARGIN_THRESHOLD", "0.05")),
            planner_cache_ttl_seconds=int(os.getenv("PLANNER_CACHE_TTL_SECONDS", "300")),
            parent_backfill=os.getenv("RETRIEVAL_PARENT_BACKFILL", "true").lower() == "true",
            original_query_rrf_weight=float(os.getenv("ORIGINAL_QUERY_RRF_WEIGHT", "2.0")),
            dense_rrf_weight=float(os.getenv("DENSE_RRF_WEIGHT", "1.0")),
            sparse_rrf_weight=float(os.getenv("SPARSE_RRF_WEIGHT", "1.0")),
        ),
        chunking=ChunkingSettings(
            parent_tokens=int(os.getenv("CHUNK_PARENT_TOKENS", "1500")),
            child_tokens=int(os.getenv("CHUNK_CHILD_TOKENS", "400")),
            overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "200")),
            version=os.getenv("CHUNK_VERSION", "semantic-v2"),
        ),
        inference=InferenceSettings(
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            reranker_model_name=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large"),
            reranker_enabled=os.getenv("RERANKER_ENABLED", "true").lower() == "true",
            embedding_service_url=(os.getenv("EMBEDDING_SERVICE_URL") or "").strip() or None,
            reranker_service_url=(os.getenv("RERANKER_SERVICE_URL") or "").strip() or None,
            request_timeout_seconds=int(os.getenv("INFERENCE_REQUEST_TIMEOUT_SECONDS", "60")),
            prefer_local_fallback=os.getenv("PREFER_LOCAL_INFERENCE_FALLBACK", "true").lower() == "true",
            nli_model_name=os.getenv(
                "NLI_MODEL_NAME",
                "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            ),
            nli_service_url=(os.getenv("NLI_SERVICE_URL") or "").strip() or None,
            nli_entailment_threshold=float(os.getenv("NLI_ENTAILMENT_THRESHOLD", "0.75")),
            nli_enabled=os.getenv("NLI_ENABLED", "true").lower() == "true",
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            embedding_max_length=int(os.getenv("EMBEDDING_MAX_LENGTH", "2048")),
        ),
        storage=StorageSettings(
            database_url=(os.getenv("DATABASE_URL") or "").strip() or None,
            sqlite_path=(os.getenv("SQLITE_PATH") or "").strip() or None,
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
    _validate_deployment(settings)
    return settings


def _validate_deployment(settings: Settings) -> None:
    if settings.deployment.mode == "host":
        return

    errors: list[str] = []
    roles = (
        settings.llm_roles.planner,
        settings.llm_roles.draft,
        settings.llm_roles.auditor,
        settings.llm_roles.judge,
    )
    if settings.deployment.mode == "api-cpu":
        for role in roles:
            if role.provider != "openai":
                errors.append("all LLM roles must use provider openai")
            endpoint = urlparse(role.base_url or "")
            try:
                port = endpoint.port
                valid_port = port in {None, 443}
            except ValueError:
                valid_port = False
            if not (
                endpoint.scheme == "https"
                and endpoint.hostname == "api.deepseek.com"
                and valid_port
                and endpoint.username is None
                and endpoint.password is None
            ):
                errors.append(
                    "all LLM roles require a secure DeepSeek HTTPS endpoint"
                )
            if not role.api_key:
                errors.append("all LLM roles require an API key")
        cpu_services = (
            ("EMBEDDING_SERVICE_URL", settings.inference.embedding_service_url, "embedding-cpu"),
            ("NLI_SERVICE_URL", settings.inference.nli_service_url, "nli-cpu"),
        )
        for name, service_url, expected_host in cpu_services:
            if urlparse(service_url or "").hostname != expected_host:
                errors.append(f"{name} must use hostname {expected_host}")
        if not settings.inference.nli_enabled:
            errors.append("NLI_ENABLED must be true")
        if settings.inference.reranker_enabled:
            reranker_host = urlparse(settings.inference.reranker_service_url or "").hostname
            if reranker_host != "reranker-cpu":
                errors.append("RERANKER_SERVICE_URL must use hostname reranker-cpu")
    else:
        expected_hosts = ("planner-llm", "generation-llm", "generation-llm", "generation-llm")
        for role, expected_host in zip(roles, expected_hosts):
            if role.provider != "vllm":
                errors.append("all LLM roles must use provider vllm")
            if urlparse(role.base_url or "").hostname != expected_host:
                errors.append(f"LLM role URL must use hostname {expected_host}")
        for name, service_url, expected_host in (
            ("EMBEDDING_SERVICE_URL", settings.inference.embedding_service_url, "embedding"),
            ("RERANKER_SERVICE_URL", settings.inference.reranker_service_url, "reranker"),
            ("NLI_SERVICE_URL", settings.inference.nli_service_url, "nli"),
        ):
            if urlparse(service_url or "").hostname != expected_host:
                errors.append(f"{name} must use hostname {expected_host}")
        if not settings.inference.nli_enabled:
            errors.append("NLI_ENABLED must be true")

    database = urlparse(settings.storage.database_url or "")
    if database.scheme not in {"postgres", "postgresql", "postgresql+psycopg"} or database.hostname in {None, "localhost", "127.0.0.1"}:
        errors.append("DATABASE_URL must use non-local PostgreSQL")
    if settings.milvus.host != "milvus":
        errors.append("MILVUS_HOST must be milvus")
    if settings.retrieval.max_audit_iterations <= 0:
        errors.append("MAX_AUDIT_ITERATIONS must be greater than zero")
    if settings.inference.prefer_local_fallback:
        errors.append("PREFER_LOCAL_INFERENCE_FALLBACK must be false in container modes")

    if errors:
        raise ConfigurationError("; ".join(dict.fromkeys(errors)))


def get_anthropic_client(
    settings: Settings | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: int | None = None,
) -> "Anthropic":
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
        api_key=api_key or resolved.anthropic_api_key,
        base_url=base_url or resolved.anthropic_base_url or None,
        http_client=httpx.Client(
            transport=_CleanTransport(),
            timeout=timeout_seconds or resolved.inference.request_timeout_seconds,
        ),
    )


__all__ = [
    "AppSettings",
    "ConfigurationError",
    "DeploymentSettings",
    "InferenceSettings",
    "LLMSettings",
    "LLMRoleSettings",
    "LLMRolesSettings",
    "MilvusSettings",
    "PathSettings",
    "RetrievalSettings",
    "ChunkingSettings",
    "StorageSettings",
    "Settings",
    "configure_logging",
    "ensure_sqlite_compatibility",
    "get_anthropic_client",
    "get_settings",
    "load_environment",
    "resolve_project_root",
]
