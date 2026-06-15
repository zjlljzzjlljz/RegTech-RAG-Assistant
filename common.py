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
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional, Tuple

import anthropic
from dotenv import load_dotenv

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

_anthropic_client: Optional[anthropic.Anthropic] = None



def configure_logging() -> None:
    """配置标准日志，输出时间戳、等级和消息内容。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )



def log_sqlite_runtime() -> None:
    """记录当前 SQLite 后端与版本，便于排查本地兼容性问题。"""
    logger.info("Active sqlite backend: %s", ACTIVE_SQLITE_BACKEND)
    logger.info("Active sqlite version: %s", ACTIVE_SQLITE_VERSION)



def load_environment(base_dir: Path) -> None:
    """从项目根目录加载 .env 文件。"""
    env_path = base_dir / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info("Environment variables loaded from: %s", env_path)



def resolve_base_dir() -> Path:
    """解析项目根目录。"""
    return Path(__file__).resolve().parent



def resolve_db_dir(base_dir: Path) -> Path:
    """解析 ChromaDB 持久化目录。"""
    return (base_dir / "chroma_db").resolve()



def resolve_project_paths() -> Tuple[Path, Path, Path]:
    """统一解析项目根目录、数据目录和 ChromaDB 目录。"""
    base_dir = resolve_base_dir()
    data_dir = (base_dir / "data").resolve()
    db_dir = resolve_db_dir(base_dir)
    return base_dir, data_dir, db_dir



def get_configured_anthropic_base_url() -> str:
    """返回 .env 中配置的自定义 Anthropic 网关地址（不含 cc-switch 本地代理）。"""
    return (os.getenv("ANTHROPIC_BASE_URL") or "").strip()



def get_effective_anthropic_endpoint() -> str:
    """返回当前实际使用的 Anthropic 端点，便于在 CLI 或 UI 中展示。"""
    configured = get_configured_anthropic_base_url()
    auth_token = (os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
    if configured and auth_token == "PROXY_MANAGED":
        return f"cc-switch local proxy → {configured}"
    return configured or "https://api.anthropic.com"



def get_anthropic_client() -> anthropic.Anthropic:
    """
    创建 Anthropic SDK 客户端。

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

    说明：
    - 不在该共享函数中添加任何 Streamlit 装饰器，以便 Streamlit 文件可按需
      使用 @st.cache_resource 包裹本函数而不改变其调用签名。
    - 保留与 10_app.py 相同的函数签名与内部逻辑分支。
    """
    global _anthropic_client

    if _anthropic_client is not None:
        return _anthropic_client

    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    base_url = (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
    auth_token = (os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()

    # 模式 1：cc-switch 本地代理，base_url 为本地端口，auth_token 为 PROXY_MANAGED
    if base_url and auth_token == "PROXY_MANAGED":
        logger.info("Detected cc-switch local proxy at %s", base_url)
        _anthropic_client = anthropic.Anthropic(auth_token=auth_token, base_url=base_url)
        return _anthropic_client

    # 模式 2：标准自定义网关（base_url 存在且 auth_token 不是 PROXY_MANAGED）
    if base_url:
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Please add it to your .env file when using a custom ANTHROPIC_BASE_URL."
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        return _anthropic_client

    # 模式 3：官方直连
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Please add it to your .env file."
        )
    _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client



def display_page(page_value: Any) -> str:
    """将底层页码转换为更适合业务用户阅读的展示值。"""
    if isinstance(page_value, int):
        return str(page_value + 1)

    if isinstance(page_value, str) and page_value.isdigit():
        return str(int(page_value) + 1)

    return str(page_value)



def preview_text(text: str, limit: int = 120) -> str:
    """生成单行文本预览，适用于 CLI、表格和日志摘要输出。"""
    normalized = " ".join((text or "").strip().split())
    if len(normalized) > limit:
        normalized = f"{normalized[:limit]}..."
    return normalized



def main() -> int:
    """运行共享模块的本地诊断，验证环境加载、SQLite 状态和 Anthropic 客户端初始化。"""
    configure_logging()

    try:
        base_dir = resolve_base_dir()
        load_environment(base_dir)
        log_sqlite_runtime()

        print(f"Base Directory: {base_dir}")
        print(f"Effective Anthropic Endpoint: {get_effective_anthropic_endpoint()}")

        try:
            _ = get_anthropic_client()
            print("Anthropic Client: OK")
        except Exception as exc:  # noqa: BLE001
            print(f"Anthropic Client: FAILED — {exc}")

        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("common.py diagnostic failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
