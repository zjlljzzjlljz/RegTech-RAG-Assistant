import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import anthropic
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-fable-5")
DEFAULT_TOP_K = 3
MAX_AUDIT_ITERATIONS = 3
TEST_QUERIES = [
    "What are the customer due diligence requirements?",
    "What must an AI do to verify a customer's identity?",
    "risk assessment procedures for money laundering",
]
DRAFTER_SYSTEM_PROMPT = (
    "You are a compliance report drafter. Write a compliance assessment based "
    "SOLELY on the evidence below. For EVERY factual claim, cite the source with "
    "[Source: filename, Page: X]. Do not invent any regulation, statute number, or "
    "requirement that does not appear in the evidence. If evidence is insufficient, "
    "state this limitation explicitly."
)
AUDITOR_SYSTEM_PROMPT = (
    "You are a strict compliance auditor. Audit the draft below. Check:\n"
    "(a) Does EVERY factual claim have an explicit source citation with\n"
    "    filename AND page number?\n"
    "(b) Does any claim reference regulations or requirements NOT in the\n"
    "    provided evidence? (hallucination check)\n"
    "(c) Is the draft missing relevant information that IS in the evidence?\n"
    "Respond with EXACTLY one word — APPROVED or REJECTED — on the first line.\n"
    "If REJECTED, provide specific, actionable feedback on subsequent lines."
)

logger = logging.getLogger(__name__)
_cross_encoder_module: Optional[Any] = None
_anthropic_client: Optional[anthropic.Anthropic] = None
_runtime_db_dir: Optional[Path] = None
_environment_loaded = False


class AuditState(TypedDict):
    messages: List[dict]
    user_query: str
    evidence: List[dict]
    compliance_draft: str
    audit_feedback: str
    audit_approved: bool
    iteration_count: int



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
    """从项目根目录加载 .env，确保脚本可独立运行。"""
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



def load_cross_encoder_module() -> Any:
    """加载 6_cross_encoder_retrieval.py，复用重排检索逻辑。"""
    global _cross_encoder_module

    if _cross_encoder_module is None:
        module_path = resolve_base_dir() / "6_cross_encoder_retrieval.py"
        _cross_encoder_module = load_module(module_path=module_path, module_name="cross_encoder_retrieval_module")

    return _cross_encoder_module



def get_cross_encoder_retrieve():
    """返回 cross_encoder_retrieve 函数引用。"""
    module = load_cross_encoder_module()
    cross_encoder_retrieve = getattr(module, "cross_encoder_retrieve", None)
    if cross_encoder_retrieve is None:
        raise AttributeError("6_cross_encoder_retrieval.py does not define cross_encoder_retrieve")
    return cross_encoder_retrieve



def get_anthropic_client() -> anthropic.Anthropic:
    """按照项目现有环境规则创建 Anthropic 客户端。"""
    global _anthropic_client

    if _anthropic_client is not None:
        return _anthropic_client

    ensure_environment_loaded()
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    auth_token = (os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()

    if base_url and auth_token == "PROXY_MANAGED":
        logger.info("Detected cc-switch local proxy at %s", base_url)
        _anthropic_client = anthropic.Anthropic(auth_token=auth_token, base_url=base_url)
        return _anthropic_client

    if base_url:
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Please add it to your .env file when using a custom ANTHROPIC_BASE_URL."
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        return _anthropic_client

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Please add it to your .env file.")

    _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client



def extract_text_from_message(response: Any) -> str:
    """从 Anthropic Messages 响应中提取文本。"""
    texts: List[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            block_text = getattr(block, "text", "")
            if block_text:
                texts.append(block_text)
    return "\n".join(texts).strip()



def preview_text(text: str, limit: int = 500) -> str:
    """生成单段预览，供 CLI 输出。"""
    normalized = (text or "").strip()
    if len(normalized) > limit:
        normalized = f"{normalized[:limit]}..."
    return normalized



def render_evidence_context(evidence: List[dict]) -> str:
    """将检索证据渲染成结构化上下文。"""
    if not evidence:
        return "No evidence was retrieved from the local ChromaDB."

    blocks: List[str] = []
    for item in evidence:
        blocks.append(
            "\n".join(
                [
                    f"{item.get('label', 'E?')}",
                    f"Source: {item.get('source_file', 'unknown')}",
                    f"Page: {item.get('page_display', 'N/A')}",
                    "Excerpt:",
                    item.get("text", ""),
                ]
            )
        )

    return "\n\n".join(blocks)



def build_drafter_system_prompt(audit_feedback: str) -> str:
    """构建 draftee 节点系统提示词。"""
    if not audit_feedback.strip():
        return DRAFTER_SYSTEM_PROMPT

    return (
        f"{DRAFTER_SYSTEM_PROMPT}\n\n"
        "You are revising a previously rejected draft. Address the auditor feedback below fully, "
        "while remaining strictly grounded in the provided evidence.\n\n"
        f"Auditor Feedback:\n{audit_feedback.strip()}"
    )



def call_claude(system_prompt: str, user_content: str, max_tokens: int) -> str:
    """执行一次非流式 Claude 调用。"""
    client = get_anthropic_client()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
    )
    return extract_text_from_message(response)



def retrieve_node(state: AuditState) -> AuditState:
    """调用 Cross-Encoder 检索，填充 top-3 evidence。"""
    next_state = dict(state)
    messages = list(next_state.get("messages", []))
    next_state["messages"] = messages

    try:
        if _runtime_db_dir is None:
            raise RuntimeError("Runtime db_dir is not initialized before retrieve_node execution.")

        cross_encoder_retrieve = get_cross_encoder_retrieve()
        evidence = cross_encoder_retrieve(query=next_state["user_query"], db_dir=_runtime_db_dir, top_k=DEFAULT_TOP_K)
        next_state["evidence"] = evidence
        messages.append(
            {
                "node": "retrieve_node",
                "status": "ok",
                "evidence_count": len(evidence),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("retrieve_node failed: %s", exc)
        next_state["evidence"] = []
        messages.append(
            {
                "node": "retrieve_node",
                "status": "error",
                "message": str(exc),
            }
        )

    return next_state



def draftee_node(state: AuditState) -> AuditState:
    """基于证据生成合规评估草稿；若被拒绝，则按审计反馈重写。"""
    next_state = dict(state)
    messages = list(next_state.get("messages", []))
    next_state["messages"] = messages
    next_state["iteration_count"] = int(next_state.get("iteration_count", 0)) + 1

    evidence_context = render_evidence_context(next_state.get("evidence", []))
    system_prompt = build_drafter_system_prompt(next_state.get("audit_feedback", ""))
    user_content = "\n\n".join(
        [
            f"User Query:\n{next_state['user_query']}",
            f"Evidence:\n{evidence_context}",
        ]
    )

    try:
        compliance_draft = call_claude(system_prompt=system_prompt, user_content=user_content, max_tokens=1000)
        if not compliance_draft:
            compliance_draft = (
                "The drafting model returned no content. Evidence may be insufficient or the model response was empty."
            )
        next_state["compliance_draft"] = compliance_draft
        messages.append(
            {
                "node": "draftee_node",
                "status": "ok",
                "iteration": next_state["iteration_count"],
                "draft_preview": preview_text(compliance_draft, limit=180),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("draftee_node failed: %s", exc)
        next_state["compliance_draft"] = f"Draft generation failed: {exc}"
        messages.append(
            {
                "node": "draftee_node",
                "status": "error",
                "iteration": next_state["iteration_count"],
                "message": str(exc),
            }
        )

    return next_state



def auditor_node(state: AuditState) -> AuditState:
    """审计当前草稿，决定是否批准或给出可执行反馈。"""
    next_state = dict(state)
    messages = list(next_state.get("messages", []))
    next_state["messages"] = messages

    evidence_context = render_evidence_context(next_state.get("evidence", []))
    user_content = "\n\n".join(
        [
            f"Original User Query:\n{next_state['user_query']}",
            f"Evidence:\n{evidence_context}",
            f"Draft To Audit:\n{next_state.get('compliance_draft', '')}",
        ]
    )

    try:
        audit_response = call_claude(system_prompt=AUDITOR_SYSTEM_PROMPT, user_content=user_content, max_tokens=300)
        first_line = audit_response.splitlines()[0].strip().upper() if audit_response else ""
        approved = first_line == "APPROVED"

        next_state["audit_approved"] = approved
        next_state["audit_feedback"] = "" if approved else (audit_response or "REJECTED\nNo actionable auditor feedback returned.")
        messages.append(
            {
                "node": "auditor_node",
                "status": "ok",
                "decision": "APPROVED" if approved else "REJECTED",
                "feedback_preview": preview_text(next_state["audit_feedback"], limit=180),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("auditor_node failed: %s", exc)
        next_state["audit_approved"] = False
        next_state["audit_feedback"] = f"Auditor call failed: {exc}"
        messages.append(
            {
                "node": "auditor_node",
                "status": "error",
                "message": str(exc),
            }
        )

    return next_state



def route_after_audit(state: AuditState) -> str:
    """根据审计结果决定结束还是继续重写。"""
    if state.get("audit_approved") or int(state.get("iteration_count", 0)) >= MAX_AUDIT_ITERATIONS:
        return "end"
    return "redraft"



def build_audit_graph():
    """构建并编译双代理合规审计状态机。"""
    workflow = StateGraph(AuditState)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("draftee_node", draftee_node)
    workflow.add_node("auditor_node", auditor_node)

    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "draftee_node")
    workflow.add_edge("draftee_node", "auditor_node")
    workflow.add_conditional_edges(
        "auditor_node",
        route_after_audit,
        {
            "end": END,
            "redraft": "draftee_node",
        },
    )
    return workflow.compile()



def run_audit(user_query: str, db_dir: Path) -> Dict[str, Any]:
    """执行完整双代理审计流程，返回最终结果与审计轨迹。"""
    global _runtime_db_dir

    _runtime_db_dir = db_dir
    graph = build_audit_graph()
    initial_state: AuditState = {
        "messages": [],
        "user_query": user_query,
        "evidence": [],
        "compliance_draft": "",
        "audit_feedback": "",
        "audit_approved": False,
        "iteration_count": 0,
    }
    final_state = graph.invoke(initial_state)

    return {
        "user_query": final_state.get("user_query", user_query),
        "evidence": final_state.get("evidence", []),
        "final_draft": final_state.get("compliance_draft", ""),
        "audit_feedback": final_state.get("audit_feedback", ""),
        "audit_trail": final_state.get("messages", []),
        "iteration_count": final_state.get("iteration_count", 0),
        "approved_status": final_state.get("audit_approved", False),
    }



def print_audit_report(result: Dict[str, Any]) -> None:
    """打印单个查询的审计结果摘要。"""
    separator = "=" * 100
    print(separator)
    print(f'QUERY: "{result["user_query"]}"')
    print(separator)
    print(f"Iteration Count: {result['iteration_count']}")
    print(f"Approved Status: {result['approved_status']}")
    print("Audit Trail:")
    for entry in result.get("audit_trail", []):
        node = entry.get("node", "unknown")
        status = entry.get("status", "unknown")
        details = []
        for key in ["evidence_count", "iteration", "decision", "message", "draft_preview", "feedback_preview"]:
            value = entry.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
        detail_text = " | ".join(details) if details else "no details"
        print(f"- {node} | {status} | {detail_text}")
    print("-" * 100)
    print("Final Draft (first 500 chars):")
    print(preview_text(result.get("final_draft", ""), limit=500) or "(empty)")
    print(separator)



def main() -> int:
    """执行三条验证查询，展示 LangGraph 双代理审计流程。"""
    configure_logging()

    try:
        ensure_environment_loaded()
        db_dir = resolve_db_dir()
        logger.info("Resolved ChromaDB directory: %s", db_dir)

        for query in TEST_QUERIES:
            try:
                result = run_audit(user_query=query, db_dir=db_dir)
                print_audit_report(result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("run_audit failed for query '%s': %s", query, exc)
                print("=" * 100)
                print(f'QUERY: "{query}"')
                print("ERROR: run_audit failed but execution will continue.")
                print(str(exc))
                print("=" * 100)

        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure while running LangGraph audit agent: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
