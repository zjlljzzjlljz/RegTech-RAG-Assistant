from __future__ import annotations

import os

import pytest

import config.settings as settings_module
from config.settings import ConfigurationError, get_settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    original = os.environ.copy()
    monkeypatch.setattr(settings_module, "resolve_project_root", lambda: tmp_path)
    get_settings.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(original)
    get_settings.cache_clear()


def _set_api_cpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "DEPLOYMENT_MODE": "api-cpu",
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://api.deepseek.com/v1",
        "LLM_API_KEY": "test-key",
        "EMBEDDING_SERVICE_URL": "http://embedding-cpu:8000",
        "NLI_ENABLED": "true",
        "NLI_SERVICE_URL": "http://nli-cpu:8000",
        "DATABASE_URL": "postgresql://user:pass@postgres:5432/app",
        "MILVUS_HOST": "milvus",
        "MAX_AUDIT_ITERATIONS": "1",
        "RERANKER_ENABLED": "false",
        "PREFER_LOCAL_INFERENCE_FALLBACK": "false",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _set_gpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "DEPLOYMENT_MODE": "gpu-self-hosted",
        "LLM_PROVIDER": "vllm",
        "PLANNER_LLM_BASE_URL": "http://planner-llm:8000/v1",
        "DRAFT_LLM_BASE_URL": "http://generation-llm:8000/v1",
        "AUDITOR_LLM_BASE_URL": "http://generation-llm:8000/v1",
        "JUDGE_LLM_BASE_URL": "http://generation-llm:8000/v1",
        "EMBEDDING_SERVICE_URL": "http://embedding:8000",
        "RERANKER_ENABLED": "true",
        "RERANKER_SERVICE_URL": "http://reranker:8000",
        "NLI_ENABLED": "true",
        "NLI_SERVICE_URL": "http://nli:8000",
        "DATABASE_URL": "postgresql://user:pass@postgres:5432/app",
        "MILVUS_HOST": "milvus",
        "MAX_AUDIT_ITERATIONS": "1",
        "PREFER_LOCAL_INFERENCE_FALLBACK": "false",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("LLM_API_KEY", raising=False)


def test_deployment_mode_defaults_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    get_settings.cache_clear()

    assert get_settings().deployment.mode == "host"


def test_invalid_deployment_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_MODE", "serverless")
    get_settings.cache_clear()

    with pytest.raises(ConfigurationError, match="DEPLOYMENT_MODE"):
        get_settings()


def test_invalid_deployment_mode_from_project_env_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    (tmp_path / ".env").write_text("DEPLOYMENT_MODE=serverless\n")

    with pytest.raises(ConfigurationError, match="DEPLOYMENT_MODE"):
        get_settings()


@pytest.mark.parametrize("mode", ["api-cpu", "gpu-self-hosted"])
def test_project_env_cannot_activate_container_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path, mode: str
) -> None:
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    (tmp_path / ".env").write_text(f"DEPLOYMENT_MODE={mode}\n")

    with pytest.raises(
        ConfigurationError,
        match="container deployment modes must be set explicitly",
    ):
        get_settings()


def test_host_loads_env_without_overriding_process_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    env_path = tmp_path / "test.env"
    env_path.write_text(
        "EMBEDDING_MODEL_NAME=from-dotenv\n"
        "DATABASE_URL=postgresql://dotenv/db\n"
    )
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "from-process")
    monkeypatch.setenv("DATABASE_URL", "postgresql://process/db")

    settings_module.load_environment(env_path=env_path, mode="host")

    assert os.environ["EMBEDDING_MODEL_NAME"] == "from-process"
    assert os.environ["DATABASE_URL"] == "postgresql://process/db"


@pytest.mark.parametrize("mode", ["api-cpu", "gpu-self-hosted"])
def test_container_modes_do_not_load_project_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path, mode: str
) -> None:
    env_path = tmp_path / "test.env"
    env_path.write_text("EMBEDDING_MODEL_NAME=from-dotenv\n")
    monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)

    settings_module.load_environment(env_path=env_path, mode=mode)

    assert "EMBEDDING_MODEL_NAME" not in os.environ


def test_host_uses_temporary_env_when_process_value_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    (tmp_path / ".env").write_text("EMBEDDING_MODEL_NAME=from-dotenv\n")
    monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)

    assert get_settings().inference.embedding_model_name == "from-dotenv"


def test_role_values_override_global_and_cache_clear_reparses(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://global.example/v1")
    monkeypatch.setenv("PLANNER_LLM_BASE_URL", "http://planner-llm:8000/v1")

    first = get_settings()
    monkeypatch.setenv("PLANNER_LLM_BASE_URL", "http://planner-llm:9000/v1")
    assert get_settings() is first

    get_settings.cache_clear()
    reparsed = get_settings()
    assert reparsed.llm_roles.planner.base_url == "http://planner-llm:9000/v1"
    assert reparsed.llm_roles.draft.base_url == "https://global.example/v1"


def test_vm_role_url_is_not_overwritten_by_dotenv_global_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    (tmp_path / ".env").write_text(
        "LLM_BASE_URL=https://api.deepseek.com/v1\n"
        "PLANNER_LLM_BASE_URL=http://planner-llm:8000/v1\n"
    )
    monkeypatch.setenv("PLANNER_LLM_BASE_URL", "http://planner-llm:9000/v1")

    assert get_settings().llm_roles.planner.base_url == "http://planner-llm:9000/v1"


def test_api_cpu_rejects_missing_embedding_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_api_cpu_environment(monkeypatch)
    monkeypatch.delenv("EMBEDDING_SERVICE_URL")

    with pytest.raises(ConfigurationError, match="EMBEDDING_SERVICE_URL"):
        get_settings()


def test_api_cpu_accepts_complete_configuration_without_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_cpu_environment(monkeypatch)

    settings = get_settings()

    assert settings.deployment.mode == "api-cpu"
    assert settings.inference.reranker_enabled is False


@pytest.mark.parametrize(
    "name",
    [
        "PLANNER_LLM_API_KEY",
        "DRAFT_LLM_API_KEY",
        "AUDITOR_LLM_API_KEY",
        "JUDGE_LLM_API_KEY",
    ],
)
def test_api_cpu_rejects_empty_role_api_key(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    _set_api_cpu_environment(monkeypatch)
    monkeypatch.setenv(name, "")

    with pytest.raises(ConfigurationError, match="all LLM roles require an API key"):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LLM_PROVIDER", "anthropic", "provider openai"),
        (
            "LLM_BASE_URL",
            "http://generation-llm:8000/v1",
            "secure DeepSeek HTTPS endpoint",
        ),
        ("LLM_API_KEY", "", "API key"),
        ("NLI_ENABLED", "false", "NLI_ENABLED"),
        ("NLI_SERVICE_URL", "", "NLI_SERVICE_URL"),
        ("DATABASE_URL", "postgresql://user:pass@localhost/app", "DATABASE_URL"),
        ("MILVUS_HOST", "localhost", "MILVUS_HOST"),
        ("MAX_AUDIT_ITERATIONS", "0", "MAX_AUDIT_ITERATIONS"),
    ],
)
def test_api_cpu_rejects_invalid_required_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    _set_api_cpu_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


@pytest.mark.parametrize(
    "url",
    [
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com:443/v1",
    ],
)
def test_api_cpu_accepts_secure_deepseek_urls(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    _set_api_cpu_environment(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", url)

    assert get_settings().llm_roles.planner.base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://api.deepseek.com/v1",
        "https://api.deepseek.com:8443/v1",
        "https://api.deepseek.com:invalid/v1",
        "https://user:password@api.deepseek.com/v1",
        "https://api.deepseek.com.example/v1",
        "api.deepseek.com/v1",
        "",
    ],
)
def test_api_cpu_rejects_insecure_deepseek_urls_without_leaking_secrets(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    _set_api_cpu_environment(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", url)
    monkeypatch.setenv("LLM_API_KEY", "dummy-not-a-real-key")

    with pytest.raises(ConfigurationError) as exc_info:
        get_settings()

    message = str(exc_info.value)
    assert message == "all LLM roles require a secure DeepSeek HTTPS endpoint"
    if url:
        assert url not in message
    assert "user" not in message
    assert "password" not in message
    assert "dummy-not-a-real-key" not in message


def test_api_cpu_requires_reranker_url_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_api_cpu_environment(monkeypatch)
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.delenv("RERANKER_SERVICE_URL", raising=False)

    with pytest.raises(ConfigurationError, match="RERANKER_SERVICE_URL"):
        get_settings()


def test_gpu_mode_requires_expected_internal_llm_hosts_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpu_environment(monkeypatch)
    settings = get_settings()
    assert settings.llm_roles.planner.api_key is None

    get_settings.cache_clear()
    monkeypatch.setenv("PLANNER_LLM_BASE_URL", "http://generation-llm:8000/v1")
    with pytest.raises(ConfigurationError, match="planner-llm"):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LLM_PROVIDER", "openai", "provider vllm"),
        ("DRAFT_LLM_BASE_URL", "http://planner-llm:8000/v1", "generation-llm"),
        ("EMBEDDING_SERVICE_URL", "", "EMBEDDING_SERVICE_URL"),
        ("RERANKER_SERVICE_URL", "", "RERANKER_SERVICE_URL"),
        ("NLI_SERVICE_URL", "", "NLI_SERVICE_URL"),
        ("NLI_ENABLED", "false", "NLI_ENABLED"),
        ("DATABASE_URL", "sqlite:///app.db", "DATABASE_URL"),
        ("MILVUS_HOST", "127.0.0.1", "MILVUS_HOST"),
        ("MAX_AUDIT_ITERATIONS", "-1", "MAX_AUDIT_ITERATIONS"),
    ],
)
def test_gpu_mode_rejects_invalid_required_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    _set_gpu_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        get_settings()


def test_host_mode_remains_permissive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_MODE", "host")
    monkeypatch.setenv("PREFER_LOCAL_INFERENCE_FALLBACK", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("NLI_SERVICE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    settings = get_settings()

    assert settings.deployment.mode == "host"
    assert settings.inference.prefer_local_fallback is True
    assert settings.llm_roles.planner.api_key is None


def test_gpu_mode_accepts_all_roles_without_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpu_environment(monkeypatch)
    for role in ("PLANNER", "DRAFT", "AUDITOR", "JUDGE"):
        monkeypatch.delenv(f"{role}_LLM_API_KEY", raising=False)

    settings = get_settings()

    assert all(
        role.api_key is None
        for role in (
            settings.llm_roles.planner,
            settings.llm_roles.draft,
            settings.llm_roles.auditor,
            settings.llm_roles.judge,
        )
    )


@pytest.mark.parametrize("mode", ["api-cpu", "gpu-self-hosted"])
def test_container_modes_reject_local_inference_fallback(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    if mode == "api-cpu":
        _set_api_cpu_environment(monkeypatch)
    else:
        _set_gpu_environment(monkeypatch)
    monkeypatch.setenv("PREFER_LOCAL_INFERENCE_FALLBACK", "true")

    with pytest.raises(ConfigurationError, match="PREFER_LOCAL_INFERENCE_FALLBACK"):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "expected_host"),
    [
        ("EMBEDDING_SERVICE_URL", "https://models.example/embedding", "embedding"),
        ("RERANKER_SERVICE_URL", "https://models.example/reranker", "reranker"),
        ("NLI_SERVICE_URL", "https://models.example/nli", "nli"),
    ],
)
def test_gpu_mode_rejects_external_inference_urls(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, expected_host: str
) -> None:
    _set_gpu_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=rf"hostname {expected_host}(?:;|$)"):
        get_settings()


@pytest.mark.parametrize("mode", ["api-cpu", "gpu-self-hosted"])
def test_container_modes_accept_sqlalchemy_psycopg_dsn(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    if mode == "api-cpu":
        _set_api_cpu_environment(monkeypatch)
    else:
        _set_gpu_environment(monkeypatch)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://user:pass@postgres:5432/app",
    )

    assert get_settings().storage.database_url == (
        "postgresql+psycopg://user:pass@postgres:5432/app"
    )


@pytest.mark.parametrize(
    ("name", "value", "expected_host"),
    [
        ("EMBEDDING_SERVICE_URL", "http://embedding-wrong:8000", "embedding-cpu"),
        ("NLI_SERVICE_URL", "http://nli-wrong:8000", "nli-cpu"),
        ("RERANKER_SERVICE_URL", "http://reranker-wrong:8000", "reranker-cpu"),
    ],
)
def test_api_cpu_rejects_wrong_internal_service_hosts(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, expected_host: str
) -> None:
    _set_api_cpu_environment(monkeypatch)
    if name == "RERANKER_SERVICE_URL":
        monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=expected_host):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "expected_host"),
    [
        ("EMBEDDING_SERVICE_URL", "http://embedding-alt:8000", "embedding"),
        ("RERANKER_SERVICE_URL", "http://reranker-alt:8000", "reranker"),
        ("NLI_SERVICE_URL", "http://nli-alt:8000", "nli"),
    ],
)
def test_gpu_mode_rejects_wrong_internal_service_hosts(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, expected_host: str
) -> None:
    _set_gpu_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=rf"hostname {expected_host}(?:;|$)"):
        get_settings()
