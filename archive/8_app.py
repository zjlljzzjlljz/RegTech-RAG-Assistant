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

import importlib.util
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import anthropic
import chromadb
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# 全局配置常量
# -----------------------------
APP_TITLE = "RegTech RAG Assistant"
COLLECTION_NAME = "regtech_parent_child_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-fable-5")
MIN_SQLITE_VERSION = (3, 35, 0)
STRICT_MODE = "Strict Grounding"
BACKGROUND_MODE = "Allow Background Knowledge"
DEFAULT_TOP_K = 4
MAX_HISTORY_MESSAGES = 12
MAX_TOKENS = 4000
TEMPERATURE = 0
MODEL_FALLBACK_STANDARD_SONNET = os.getenv("ANTHROPIC_FALLBACK_SONNET", "claude-haiku-4-5-20251001")
MODEL_FALLBACK_STANDARD_HAIKU = os.getenv("ANTHROPIC_FALLBACK_HAIKU", "claude-fable-5")

logger = logging.getLogger(__name__)
_cross_encoder_module: Optional[Any] = None
_hyde_module: Optional[Any] = None
FAST_PATH_LABEL = "Fast Path (Cross-Encoder)"
RESCUE_PATH_LABEL = "Rescue Path (HyDE + Cross-Encoder)"

SYSTEM_PROMPT = """
You are a business-formal RegTech research assistant.

You must answer regulatory and compliance questions using the retrieved evidence first.
You will receive:
1. Retrieved evidence snippets labeled E1, E2, E3...
2. An active answering mode
3. The user's question

Universal rules:
- Respond in the same language as the user's question unless the user explicitly requests another language.
- Keep the tone professional, precise, and business-formal.
- Never fabricate citations, page numbers, or source files.
- When you rely on retrieved evidence, cite it inline using the exact evidence labels, for example [E1] or [E1][E3].
- If multiple evidence chunks support the same claim, cite the most relevant labels only.
- Do not mention internal implementation details such as embeddings, vector stores, or prompt caching.

Mode rules:
- If Active Mode is STRICT_GROUNDING, answer only from retrieved evidence. If the evidence is insufficient, say clearly that the current document set does not provide enough support to confirm the answer.
- If Active Mode is ALLOW_BACKGROUND_KNOWLEDGE, still prioritize retrieved evidence. If the evidence is insufficient, you may add limited general background knowledge, but you must explicitly separate it from document-grounded findings under a heading like 'Background context (not directly supported by retrieved passages)'.

Response structure:
- Start with a concise direct answer or conclusion.
- Follow with short supporting bullets or paragraphs.
- Keep the answer readable for compliance, audit, and management stakeholders.
""".strip()


def _parse_sqlite_version(version: str) -> Tuple[int, int, int]:
    """将 sqlite3 版本字符串解析为可比较的三元组。"""
    parsed_parts: List[int] = []
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
        "or run this app with a Python build linked against a newer SQLite runtime."
    )


def configure_logging() -> None:
    """配置标准日志，便于调试 Streamlit、Chroma 与 Claude 调用链路。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )



def resolve_paths() -> Tuple[Path, Path]:
    """解析项目根目录与 ChromaDB 持久化目录。"""
    base_dir = Path(__file__).resolve().parent
    db_dir = (base_dir / "chroma_db").resolve()
    return base_dir, db_dir



def load_environment(base_dir: Path) -> None:
    """从项目根目录加载 .env 文件。"""
    env_path = base_dir / ".env"
    # 这里使用 override=True，确保 .env 中的显式配置能够覆盖空字符串环境变量。
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info("Environment variables loaded from: %s", env_path)



def log_sqlite_runtime() -> None:
    """记录当前 SQLite 后端与版本，便于排查本地兼容性问题。"""
    logger.info("Active sqlite backend: %s", ACTIVE_SQLITE_BACKEND)
    logger.info("Active sqlite version: %s", ACTIVE_SQLITE_VERSION)



def display_page(page_value: Any) -> str:
    """将底层页码转换为更适合业务用户阅读的展示值。"""
    if isinstance(page_value, int):
        return str(page_value + 1)

    if isinstance(page_value, str) and page_value.isdigit():
        return str(int(page_value) + 1)

    return str(page_value)



def build_evidence_records(documents_with_scores: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
    """将检索结果标准化为 UI 和提示词都可复用的证据结构。"""
    records: List[Dict[str, Any]] = []

    for index, (document, score) in enumerate(documents_with_scores, start=1):
        source_file = document.metadata.get("source_file", "unknown")
        page_value = document.metadata.get("page", "N/A")
        evidence_text = document.page_content.strip()

        records.append(
            {
                "label": f"E{index}",
                "source_file": source_file,
                "page": page_value,
                "page_display": display_page(page_value),
                "score": float(score),
                "text": evidence_text,
            }
        )

    return records



def build_retrieved_context(evidence_records: List[Dict[str, Any]]) -> str:
    """把检索证据渲染成结构化上下文，供 Claude 严格引用。"""
    if not evidence_records:
        return "No retrieved document evidence was found for this turn."

    blocks: List[str] = []
    for record in evidence_records:
        blocks.append(
            "\n".join(
                [
                    f"{record['label']}",
                    f"Source: {record['source_file']}",
                    f"Page: {record['page_display']}",
                    "Excerpt:",
                    record["text"],
                ]
            )
        )

    return "\n\n".join(blocks)



def build_current_turn_prompt(user_query: str, mode: str) -> str:
    """构建当前轮的动态提示词，把模式与问题显式传给模型。"""
    mode_description = (
        "STRICT_GROUNDING: Use only the retrieved document evidence. Refuse to over-claim if support is insufficient."
        if mode == STRICT_MODE
        else "ALLOW_BACKGROUND_KNOWLEDGE: Prioritize retrieved evidence, but if needed you may add limited background knowledge that is clearly separated from document-grounded findings."
    )

    return "\n".join(
        [
            f"Active Mode: {mode}",
            f"Mode Policy: {mode_description}",
            "User Question:",
            user_query,
        ]
    )



def extract_used_citation_labels(answer_text: str) -> List[str]:
    """从模型回答中提取被实际引用的证据标签。"""
    labels = set(re.findall(r"E\d+", answer_text))
    return sorted(labels, key=lambda label: int(label[1:]))



def deduplicate_citations(evidence_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 source_file + page 去重，避免 citation 列表重复。"""
    seen = set()
    deduplicated: List[Dict[str, Any]] = []

    for record in evidence_records:
        key = (record["source_file"], record["page_display"])
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(record)

    return deduplicated



def select_citations(answer_text: str, evidence_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """优先展示模型明确引用的证据；若未检测到标签，则退回检索证据页码。"""
    labels = extract_used_citation_labels(answer_text)
    if labels:
        label_set = set(labels)
        selected = [record for record in evidence_records if record["label"] in label_set]
        return deduplicate_citations(selected)

    return deduplicate_citations(evidence_records)



def render_citations_markdown(citations: List[Dict[str, Any]]) -> str:
    """把引用信息渲染为统一的 Markdown。"""
    if not citations:
        return ""

    lines = ["**Citations**"]
    for record in citations:
        lines.append(f"- `{record['label']}` — {record['source_file']}, p.{record['page_display']}")
    return "\n".join(lines)



def get_configured_anthropic_base_url() -> str:
    """返回 .env 中配置的自定义 Anthropic 网关地址（不含 cc-switch 本地代理）。"""
    return (os.getenv("ANTHROPIC_BASE_URL") or "").strip()


def get_effective_anthropic_endpoint() -> str:
    """返回当前实际使用的 Anthropic 端点，便于在 UI 中展示。"""
    configured = get_configured_anthropic_base_url()
    auth_token = (os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
    if configured and auth_token == "PROXY_MANAGED":
        return f"cc-switch local proxy → {configured}"
    return configured or "https://api.anthropic.com"



def format_anthropic_status_error(exc: anthropic.APIStatusError) -> str:
    """
    将 Anthropic SDK 的状态码异常转换为可运维、可排障的错误提示。

    目标不是简单回显 502，而是尽量区分：
    1. 本地 API Key 缺失或格式问题；
    2. 自定义网关拒绝当前请求；
    3. 自定义网关可以接收请求，但它到上游 Claude 的鉴权失败；
    4. 其他普通 API 状态码错误。
    """
    status_code = getattr(exc, "status_code", None)
    message = getattr(exc, "message", str(exc))
    body = getattr(exc, "body", None)

    error_type = ""
    upstream_message = ""
    if isinstance(body, dict):
        error_payload = body.get("error") or {}
        error_type = str(error_payload.get("type") or "")
        upstream_message = str(error_payload.get("message") or "")

    normalized_message = upstream_message.lower()
    endpoint = get_effective_anthropic_endpoint()
    has_custom_gateway = bool(get_configured_anthropic_base_url())

    if has_custom_gateway and status_code == 502 and error_type == "upstream_error":
        if "authentication failed" in normalized_message:
            return (
                "The configured Anthropic gateway accepted the request but failed to authenticate with its upstream Claude provider. "
                f"Gateway endpoint: {endpoint}. "
                "This indicates a gateway-side credential, provider-routing, or upstream account configuration problem rather than a local `.env` loading issue. "
                "Please verify the gateway's upstream Anthropic credentials, model allowlist, and account mapping, or temporarily remove `ANTHROPIC_BASE_URL` to use the official Anthropic endpoint directly."
            )

        return (
            "The configured Anthropic gateway returned a 502 upstream error while brokering the Claude request. "
            f"Gateway endpoint: {endpoint}. "
            "This usually indicates a gateway-side routing or upstream provider failure rather than a local Streamlit bug. "
            f"Raw gateway message: {upstream_message or message}"
        )

    if has_custom_gateway and status_code == 503:
        if "no available accounts" in normalized_message or "pricing restriction" in normalized_message:
            return (
                "The configured Anthropic gateway is reachable, but it currently has no upstream account capacity for the requested Claude model. "
                f"Gateway endpoint: {endpoint}. "
                "This is a gateway inventory or channel-pricing restriction rather than a local application bug. "
                f"Raw gateway message: {upstream_message or message}"
            )

        return (
            "The configured Anthropic gateway returned a 503 service availability error while brokering the Claude request. "
            f"Gateway endpoint: {endpoint}. "
            f"Raw gateway message: {upstream_message or message}"
        )

    if has_custom_gateway and status_code in {401, 403}:
        return (
            "The configured Anthropic gateway rejected the request credentials. "
            f"Gateway endpoint: {endpoint}. "
            "Please verify that the gateway expects an Anthropic `x-api-key`, that the supplied key is authorized on that gateway, and that any tenant or workspace bindings are correct."
        )

    if status_code == 401:
        return "Anthropic authentication failed. Please verify `ANTHROPIC_API_KEY`."

    if status_code == 404 and has_custom_gateway:
        return (
            "The configured Anthropic gateway did not expose the expected Anthropic Messages API route. "
            f"Gateway endpoint: {endpoint}. "
            "Please verify that the proxy is Anthropic-compatible and forwards `/v1/messages` requests correctly."
        )

    return f"Anthropic API error ({status_code}): {message}"



@st.cache_resource(show_spinner=False)
def get_anthropic_client() -> anthropic.Anthropic:
    """
    缓存 Anthropic SDK 客户端，避免每次重渲染重复初始化。

    支持三种接入模式，按优先级自动选择：

    1. cc-switch 本地代理（ANTHROPIC_BASE_URL=http://127.0.0.1:15722
       + ANTHROPIC_AUTH_TOKEN=PROXY_MANAGED）：
       cc-switch 是 macOS 上的 Claude 代理应用，会自动在环境变量中注入
       base_url 并通过 PROXY_MANAGED token 声明自己管理认证。
       SDK 客户端使用 auth_token="PROXY_MANAGED" + base_url 初始化即可。

    2. 标准自定义网关（.env 中有 ANTHROPIC_BASE_URL 且 auth_token 非 PROXY_MANAGED）：
       使用 .env 中的 api_key + base_url。

    3. 官方 Anthropic 直连（.env 中仅有 api_key，无 base_url）：
       使用 .env 中的 api_key，不传 base_url。
    """
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    auth_token = (os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()

    # 模式 1：cc-switch 本地代理，base_url 为本地端口，auth_token 为 PROXY_MANAGED
    if base_url and auth_token == "PROXY_MANAGED":
        logger.info("Detected cc-switch local proxy at %s", base_url)
        return anthropic.Anthropic(auth_token=auth_token, base_url=base_url)

    # 模式 2：标准自定义网关（base_url 存在且 auth_token 不是 PROXY_MANAGED）
    if base_url:
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Please add it to your .env file when using a custom ANTHROPIC_BASE_URL."
            )
        return anthropic.Anthropic(api_key=api_key, base_url=base_url)

    # 模式 3：官方直连
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Please add it to your .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_collection_count(db_dir: str) -> int:
    """读取目标 collection 的 chunk 数量，用于启动健康检查。"""
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name=COLLECTION_NAME)
    return int(collection.count())



def load_module(module_path: Path, module_name: str) -> Any:
    """通过 importlib 从脚本路径加载模块。"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def load_cross_encoder_module() -> Any:
    """加载 6_cross_encoder_retrieval.py。"""
    global _cross_encoder_module

    if _cross_encoder_module is None:
        module_path = Path(__file__).resolve().parent / "6_cross_encoder_retrieval.py"
        _cross_encoder_module = load_module(module_path=module_path, module_name="cross_encoder_retrieval_module")

    return _cross_encoder_module



def load_hyde_module() -> Any:
    """加载 7_hyde_retrieval.py。"""
    global _hyde_module

    if _hyde_module is None:
        module_path = Path(__file__).resolve().parent / "7_hyde_retrieval.py"
        _hyde_module = load_module(module_path=module_path, module_name="hyde_retrieval_module")

    return _hyde_module



def get_cross_encoder_retrieve():
    """返回 cross_encoder_retrieve 函数引用。"""
    module = load_cross_encoder_module()
    cross_encoder_retrieve = getattr(module, "cross_encoder_retrieve", None)
    if cross_encoder_retrieve is None:
        raise AttributeError("6_cross_encoder_retrieval.py does not define cross_encoder_retrieve")
    return cross_encoder_retrieve



def get_hyde_retrieve():
    """返回 hyde_retrieve 函数引用。"""
    module = load_hyde_module()
    hyde_retrieve = getattr(module, "hyde_retrieve", None)
    if hyde_retrieve is None:
        raise AttributeError("7_hyde_retrieval.py does not define hyde_retrieve")
    return hyde_retrieve



def retrieve_evidence_adaptive(query: str, top_k: int, db_dir: Path) -> List[Dict[str, Any]]:
    """
    优先走 Cross-Encoder 快速路径；若匹配分数不足，再触发 HyDE 救援路径。

    检索路径会写入 session_state，供 sidebar 与 audit panel 复用。
    """
    cross_encoder_retrieve = get_cross_encoder_retrieve()
    hyde_retrieve = get_hyde_retrieve()

    results = cross_encoder_retrieve(query=query, db_dir=db_dir, top_k=max(top_k, 5))
    top_ce_score = results[0]["score"] if results else -999.0

    if top_ce_score >= 0:
        st.session_state["last_retrieval_path"] = FAST_PATH_LABEL
        return results[:top_k]

    hyde_results = hyde_retrieve(query=query, db_dir=db_dir, top_k=max(top_k, 5))
    top_hyde_score = hyde_results[0]["score"] if hyde_results else float("-inf")

    if top_hyde_score > top_ce_score:
        st.session_state["last_retrieval_path"] = RESCUE_PATH_LABEL
        return hyde_results[:top_k]

    st.session_state["last_retrieval_path"] = FAST_PATH_LABEL
    return results[:top_k]



def build_api_messages(history: List[Dict[str, Any]], user_query: str, mode: str, evidence_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构建发给 Claude 的消息列表，只保留历史文本，不重复携带旧证据。"""
    messages: List[Dict[str, Any]] = []

    for item in history[-MAX_HISTORY_MESSAGES:]:
        messages.append(
            {
                "role": item["role"],
                "content": item["text"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_retrieved_context(evidence_records),
                },
                {
                    "type": "text",
                    "text": build_current_turn_prompt(user_query=user_query, mode=mode),
                },
            ],
        }
    )
    return messages



def extract_usage(final_message: anthropic.types.Message) -> Dict[str, Any]:
    """把使用量信息整理为侧边栏可展示的结构。"""
    usage = final_message.usage
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
    }



def build_generation_tiers() -> List[Dict[str, Any]]:
    """
    定义 Claude 请求的优雅降级链路。

    设计原则：
    1. 先尝试理想态（主模型 + adaptive thinking + effort）。
    2. 若不兼容高级参数，则保留主模型，仅移除高级参数。
    3. 若主模型当前不可用，再降级到备选模型。
    """
    return [
        {
            "tier_key": "tier_1_full",
            "tier_label": f"Tier 1 · {MODEL_NAME} (full)",
            "model": MODEL_NAME,
            "request_kwargs": {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
            "fallback_note": "",
        },
        {
            "tier_key": "tier_2_standard",
            "tier_label": f"Tier 2 · {MODEL_NAME} (standard)",
            "model": MODEL_NAME,
            "request_kwargs": {"temperature": TEMPERATURE},
            "fallback_note": (
                "Note: Active model degraded to standard mode because the gateway rejected advanced Claude parameters."
            ),
        },
        {
            "tier_key": "tier_3_fallback",
            "tier_label": f"Tier 3 · {MODEL_FALLBACK_STANDARD_SONNET} Fallback",
            "model": MODEL_FALLBACK_STANDARD_SONNET,
            "request_kwargs": {"temperature": TEMPERATURE},
            "fallback_note": (
                f"Note: Active model degraded to {MODEL_FALLBACK_STANDARD_SONNET} as fallback."
            ),
        },
        {
            "tier_key": "tier_3b_fallback",
            "tier_label": f"Tier 3B · {MODEL_FALLBACK_STANDARD_HAIKU} Final Fallback",
            "model": MODEL_FALLBACK_STANDARD_HAIKU,
            "request_kwargs": {"temperature": TEMPERATURE},
            "fallback_note": (
                f"Note: Active model degraded to {MODEL_FALLBACK_STANDARD_HAIKU} as final fallback."
            ),
        },
    ]



def summarize_attempt_failure(exc: anthropic.APIStatusError) -> Dict[str, Any]:
    """抽取单次失败尝试的关键信息，便于在 UI 和日志中复盘。"""
    return {
        "status_code": getattr(exc, "status_code", None),
        "message": getattr(exc, "message", str(exc)),
    }



def stream_claude_answer(
    history: List[Dict[str, Any]],
    user_query: str,
    mode: str,
    evidence_records: List[Dict[str, Any]],
) -> Tuple[Generator[str, None, None], Dict[str, Any]]:
    """
    以流式方式调用 Claude，并把最终消息、降级状态与错误状态回传给 Streamlit。

    优雅降级策略：
    - Tier 1: Opus 4.7 + adaptive thinking + effort
    - Tier 2: Opus 4.7 基础 messages 调用（移除高级参数）
    - Tier 3: 兼容性优先的 Sonnet / Haiku 基础调用
    """
    client = get_anthropic_client()
    stream_state: Dict[str, Any] = {
        "final_message": None,
        "error": None,
        "active_tier": None,
        "active_model": None,
        "fallback_note": "",
        "attempt_failures": [],
    }
    api_messages = build_api_messages(history, user_query, mode, evidence_records)
    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    generation_tiers = build_generation_tiers()

    def generator() -> Generator[str, None, None]:
        for index, tier in enumerate(generation_tiers):
            yielded_any_text = False
            try:
                with client.messages.stream(
                    model=tier["model"],
                    max_tokens=MAX_TOKENS,
                    system=system_blocks,
                    messages=api_messages,
                    **tier["request_kwargs"],
                ) as stream:
                    for text in stream.text_stream:
                        yielded_any_text = True
                        yield text

                    stream_state["final_message"] = stream.get_final_message()
                    stream_state["active_tier"] = tier["tier_label"]
                    stream_state["active_model"] = tier["model"]
                    stream_state["fallback_note"] = tier["fallback_note"]
                    return
            except anthropic.AuthenticationError:
                stream_state["error"] = "Anthropic authentication failed. Please verify `ANTHROPIC_API_KEY`."
                logger.exception("Anthropic authentication failed")
                return
            except anthropic.RateLimitError as exc:
                stream_state["error"] = f"Anthropic rate limit encountered: {exc.message}"
                logger.exception("Anthropic rate limit error")
                return
            except anthropic.APIConnectionError:
                stream_state["error"] = (
                    "Network error while connecting to Anthropic. "
                    f"Endpoint: {get_effective_anthropic_endpoint()}. Please retry."
                )
                logger.exception("Anthropic connection error")
                return
            except anthropic.APIStatusError as exc:
                failure_summary = summarize_attempt_failure(exc)
                stream_state["attempt_failures"].append(
                    {
                        "tier": tier["tier_label"],
                        "model": tier["model"],
                        **failure_summary,
                    }
                )

                is_last_tier = index == len(generation_tiers) - 1
                if yielded_any_text or is_last_tier:
                    stream_state["error"] = format_anthropic_status_error(exc)
                    logger.exception(
                        "Anthropic API status error on %s using model %s",
                        tier["tier_label"],
                        tier["model"],
                    )
                    return

                logger.warning(
                    "Claude request failed on %s using model %s with status %s; retrying next fallback tier. Message: %s",
                    tier["tier_label"],
                    tier["model"],
                    failure_summary["status_code"],
                    failure_summary["message"],
                )
            except Exception as exc:  # noqa: BLE001
                stream_state["error"] = f"Unexpected Claude runtime failure: {exc}"
                logger.exception("Unexpected Claude runtime failure")
                return

    return generator(), stream_state



def initialize_session_state() -> None:
    """初始化 Streamlit 会话状态。"""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_evidence", [])
    st.session_state.setdefault("last_usage", None)
    st.session_state.setdefault("last_mode", STRICT_MODE)
    st.session_state.setdefault("last_active_model", MODEL_NAME)
    st.session_state.setdefault("last_active_tier", f"Tier 1 · {MODEL_NAME} (full)")
    st.session_state.setdefault("last_fallback_note", "")
    st.session_state.setdefault("last_attempt_failures", [])
    st.session_state.setdefault("last_retrieval_path", FAST_PATH_LABEL)



def clear_chat_state() -> None:
    """清空聊天历史，但保留底层数据库与模型缓存。"""
    st.session_state["chat_history"] = []
    st.session_state["last_evidence"] = []
    st.session_state["last_usage"] = None
    st.session_state["last_mode"] = STRICT_MODE
    st.session_state["last_active_model"] = MODEL_NAME
    st.session_state["last_active_tier"] = f"Tier 1 · {MODEL_NAME} (full)"
    st.session_state["last_fallback_note"] = ""
    st.session_state["last_attempt_failures"] = []
    st.session_state["last_retrieval_path"] = FAST_PATH_LABEL



def render_sidebar(collection_count: int) -> Tuple[str, int]:
    """渲染侧边栏配置区。"""
    st.sidebar.header("Control Panel")
    mode = st.sidebar.radio(
        "Answer Mode",
        options=[STRICT_MODE, BACKGROUND_MODE],
        index=0,
        help="Strict Grounding 默认只允许基于检索证据作答；Allow Background Knowledge 允许在证据不足时补充一般背景知识。",
    )
    top_k = st.sidebar.slider("Top-k Retrieved Chunks", min_value=2, max_value=8, value=DEFAULT_TOP_K, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Runtime Status**")
    st.sidebar.write(f"Preferred Model: `{MODEL_NAME}`")
    st.sidebar.write(f"Active Model: `{st.session_state.get('last_active_model', MODEL_NAME)}`")
    st.sidebar.write(f"Active Tier: `{st.session_state.get('last_active_tier', f'Tier 1 · {MODEL_NAME} (full)')}`")
    st.sidebar.write(f"Collection: `{COLLECTION_NAME}`")
    st.sidebar.write(f"Chunk Count: `{collection_count}`")
    st.sidebar.write(f"Retrieval Path: `{st.session_state.get('last_retrieval_path', FAST_PATH_LABEL)}`")
    st.sidebar.write(f"SQLite: `{ACTIVE_SQLITE_VERSION}`")
    st.sidebar.write(f"Anthropic Endpoint: `{get_effective_anthropic_endpoint()}`")

    fallback_note = st.session_state.get("last_fallback_note") or ""
    if fallback_note:
        st.sidebar.warning(fallback_note)

    attempt_failures = st.session_state.get("last_attempt_failures") or []
    if attempt_failures:
        with st.sidebar.expander("Fallback Trace", expanded=False):
            for failure in attempt_failures:
                st.write(
                    f"- {failure['tier']} | model=`{failure['model']}` | status=`{failure['status_code']}`"
                )
                st.caption(failure["message"])

    if st.session_state.get("last_usage"):
        usage = st.session_state["last_usage"]
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Last API Usage**")
        st.sidebar.write(f"Input Tokens: `{usage['input_tokens']}`")
        st.sidebar.write(f"Cache Write Tokens: `{usage['cache_creation_input_tokens']}`")
        st.sidebar.write(f"Cache Read Tokens: `{usage['cache_read_input_tokens']}`")
        st.sidebar.write(f"Output Tokens: `{usage['output_tokens']}`")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Conversation", use_container_width=True):
        clear_chat_state()
        st.rerun()

    return mode, top_k



def render_chat_history() -> None:
    """渲染已有聊天历史。"""
    for item in st.session_state["chat_history"]:
        with st.chat_message(item["role"]):
            st.markdown(item["text"])
            citations = item.get("citations") or []
            citations_markdown = render_citations_markdown(citations)
            if citations_markdown:
                st.markdown(citations_markdown)



def render_audit_panel() -> None:
    """渲染右侧 RegTech Audit Panel。"""
    st.markdown("### RegTech Audit Panel")
    st.caption("显示当前轮最后一次检索到的证据片段、来源文件与页码。")

    evidence_records = st.session_state.get("last_evidence") or []
    last_mode = st.session_state.get("last_mode") or STRICT_MODE
    last_retrieval_path = st.session_state.get("last_retrieval_path") or FAST_PATH_LABEL
    st.write(f"**Last Mode:** `{last_mode}`")
    st.write(f"**Retrieval Path:** `{last_retrieval_path}`")
    st.write(f"**Evidence Chunks:** `{len(evidence_records)}`")

    if not evidence_records:
        st.info("尚未执行检索。提交一个问题后，这里会显示最新证据。")
        return

    for record in evidence_records:
        title = f"{record['label']} | {record['source_file']} | p.{record['page_display']}"
        with st.expander(title, expanded=False):
            st.caption(f"Similarity Score: {record['score']:.4f}")
            st.write(record["text"])



def render_empty_evidence_message(mode: str, user_query: str) -> Dict[str, Any]:
    """当严格模式下完全无证据时，直接返回业务友好的拒答。"""
    if mode == STRICT_MODE:
        answer_text = (
            "I cannot confirm the answer from the currently retrieved regulatory material. "
            "Please refine the question, broaden the retrieval scope, or ingest additional documents before relying on a conclusion."
        )
    else:
        answer_text = (
            "No directly retrieved document evidence was found for this question. "
            "You may switch to Strict Grounding for a hard refusal posture, or continue in background mode for a cautiously qualified answer."
        )

    return {
        "role": "assistant",
        "text": answer_text,
        "citations": [],
        "question": user_query,
    }



def main() -> None:
    """启动 Streamlit RegTech RAG 聊天应用。"""
    configure_logging()
    base_dir, db_dir = resolve_paths()
    load_environment(base_dir)
    log_sqlite_runtime()

    st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="wide")
    st.title("📘 RegTech RAG Assistant")
    st.caption("A grounded Streamlit research workspace for regulatory Q&A with Claude, local ChromaDB, and page-level citations.")

    initialize_session_state()

    try:
        collection_count = get_collection_count(str(db_dir))
    except Exception as exc:  # noqa: BLE001
        st.error(
            "ChromaDB is not ready. Please run `make build-llamaindex` first to create the local collection `regtech_parent_child_docs`."
        )
        st.exception(exc)
        return

    try:
        _ = get_anthropic_client()
    except Exception as exc:  # noqa: BLE001
        st.error("Anthropic client initialization failed. Please verify your `.env` configuration.")
        st.exception(exc)
        return

    mode, top_k = render_sidebar(collection_count)

    chat_column, audit_column = st.columns([3, 1])

    with chat_column:
        render_chat_history()
        user_query = st.chat_input("Ask a grounded regulatory question...")

    with audit_column:
        render_audit_panel()

    if not user_query:
        return

    st.session_state["chat_history"].append({"role": "user", "text": user_query})
    st.session_state["last_mode"] = mode

    with chat_column:
        with st.chat_message("user"):
            st.markdown(user_query)

    try:
        with st.spinner("Retrieving supporting evidence from local ChromaDB..."):
            evidence_records = retrieve_evidence_adaptive(query=user_query, top_k=top_k, db_dir=db_dir)
    except Exception as exc:  # noqa: BLE001
        with chat_column:
            with st.chat_message("assistant"):
                st.error(f"Document retrieval failed: {exc}")
        logger.exception("Document retrieval failed")
        return

    st.session_state["last_evidence"] = evidence_records

    if not evidence_records and mode == STRICT_MODE:
        assistant_message = render_empty_evidence_message(mode=mode, user_query=user_query)
        st.session_state["chat_history"].append(assistant_message)
        with chat_column:
            with st.chat_message("assistant"):
                st.markdown(assistant_message["text"])
        st.rerun()

    with chat_column:
        with st.chat_message("assistant"):
            response_generator, stream_state = stream_claude_answer(
                history=st.session_state["chat_history"][:-1],
                user_query=user_query,
                mode=mode,
                evidence_records=evidence_records,
            )
            streamed_output = st.write_stream(response_generator)

            answer_text = streamed_output if isinstance(streamed_output, str) else ""

            if stream_state.get("error"):
                st.error(stream_state["error"])
                answer_text = answer_text or ""

            citations = select_citations(answer_text=answer_text, evidence_records=evidence_records)
            citations_markdown = render_citations_markdown(citations)
            if citations_markdown:
                st.markdown(citations_markdown)

    final_message = stream_state.get("final_message")
    if final_message is not None:
        st.session_state["last_usage"] = extract_usage(final_message)

    st.session_state["last_active_model"] = stream_state.get("active_model") or MODEL_NAME
    st.session_state["last_active_tier"] = stream_state.get("active_tier") or f"Tier 1 · {MODEL_NAME} (full)"
    st.session_state["last_fallback_note"] = stream_state.get("fallback_note") or ""
    st.session_state["last_attempt_failures"] = stream_state.get("attempt_failures") or []

    if stream_state.get("error") and not answer_text:
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "text": stream_state["error"],
                "citations": [],
                "mode": mode,
            }
        )
        st.rerun()

    if answer_text:
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "text": answer_text,
                "citations": citations,
                "mode": mode,
            }
        )

    st.rerun()


if __name__ == "__main__":
    main()
