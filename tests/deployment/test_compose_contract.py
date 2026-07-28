from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml


COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker-compose.yml"
COMMON_FIELDS = {
    "image",
    "working_dir",
    "command",
    "volumes",
    "networks",
    "healthcheck",
}
ROLE_PREFIXES = ("PLANNER", "DRAFT", "AUDITOR", "JUDGE")


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(COMPOSE_PATH.read_text())


@pytest.fixture(scope="module")
def services(compose):
    return compose["services"]


def _depends_are_healthy(service, expected):
    dependencies = service["depends_on"]
    assert set(dependencies) == set(expected)
    assert all(value["condition"] == "service_healthy" for value in dependencies.values())


def _healthcheck_url(service):
    return " ".join(str(part) for part in service["healthcheck"]["test"])


def test_static_infrastructure_ports_are_loopback_bound(services):
    expected = {
        "postgres": ["127.0.0.1:5432:5432"],
        "milvus": ["127.0.0.1:19530:19530", "127.0.0.1:9091:9091"],
        "minio": ["127.0.0.1:9000:9000", "127.0.0.1:9001:9001"],
        "attu": ["127.0.0.1:8000:3000"],
        "embedding-cpu": ["127.0.0.1:8101:8000"],
        "nli-cpu": ["127.0.0.1:8103:8000"],
    }
    if "reranker-cpu" in services:
        expected["reranker-cpu"] = ["127.0.0.1:8102:8000"]
    for name, ports in expected.items():
        assert services[name]["ports"] == ports


def test_static_profiles_and_dependencies_are_self_contained(services):
    assert "profiles" not in services["minio"]
    assert services["attu"]["profiles"] == ["tools"]
    assert services["streamlit-api"]["profiles"] == ["api-cpu"]
    assert services["streamlit-gpu"]["profiles"] == ["gpu-self-hosted"]
    for name in ("embedding-cpu", "nli-cpu"):
        assert services[name]["profiles"] == ["api-cpu"]
    for name in ("planner-llm", "generation-llm", "embedding", "reranker", "nli"):
        assert services[name]["profiles"] == ["gpu-self-hosted"]
    for service_name, service in services.items():
        service_profiles = service.get("profiles", [])
        for dependency in service.get("depends_on", {}):
            dependency_profiles = services[dependency].get("profiles", [])
            assert not dependency_profiles or set(service_profiles) & set(dependency_profiles), service_name


def test_static_api_cpu_environment_and_dependencies(services):
    app = services["streamlit-api"]
    env = app["environment"]
    assert env["DEPLOYMENT_MODE"] == "api-cpu"
    assert env["MILVUS_HOST"] == "milvus"
    assert urlparse(env["DATABASE_URL"]).hostname == "postgres"
    assert env["LLM_PROVIDER"] == "openai"
    for role in ROLE_PREFIXES:
        assert env[f"{role}_LLM_MODEL"]
        assert "api.deepseek.com" in env[f"{role}_LLM_BASE_URL"]
        api_key_interpolation = env[f"{role}_LLM_API_KEY"]
        assert api_key_interpolation == "${LLM_API_KEY:-}"
        assert ":?" not in api_key_interpolation
    assert env["EMBEDDING_SERVICE_URL"] == "http://embedding-cpu:8000"
    assert env["NLI_SERVICE_URL"] == "http://nli-cpu:8000"
    assert env["NLI_ENABLED"] == "true"
    assert env["RERANKER_ENABLED"] == "false"
    assert env["PREFER_LOCAL_INFERENCE_FALLBACK"] == "false"
    _depends_are_healthy(app, ("postgres", "milvus", "embedding-cpu", "nli-cpu"))
    assert not ({"planner-llm", "generation-llm", "reranker-cpu"} & set(app["depends_on"]))


def test_static_gpu_environment_and_dependencies(services):
    app = services["streamlit-gpu"]
    env = app["environment"]
    assert env["DEPLOYMENT_MODE"] == "gpu-self-hosted"
    assert env["LLM_PROVIDER"] == "vllm"
    gpu_services = (
        service
        for service in services.values()
        if "gpu-self-hosted" in service.get("profiles", [])
    )
    assert all(
        not any(key.endswith("LLM_API_KEY") for key in service.get("environment", {}))
        for service in gpu_services
    )
    assert urlparse(env["PLANNER_LLM_BASE_URL"]).hostname == "planner-llm"
    for role in ("DRAFT", "AUDITOR", "JUDGE"):
        assert urlparse(env[f"{role}_LLM_BASE_URL"]).hostname == "generation-llm"
    assert env["EMBEDDING_SERVICE_URL"] == "http://embedding:8000"
    assert env["RERANKER_SERVICE_URL"] == "http://reranker:8000"
    assert env["NLI_SERVICE_URL"] == "http://nli:8000"
    assert env["NLI_ENABLED"] == env["RERANKER_ENABLED"] == "true"
    assert env["PREFER_LOCAL_INFERENCE_FALLBACK"] == "false"
    assert env["MILVUS_HOST"] == "milvus"
    assert urlparse(env["DATABASE_URL"]).hostname == "postgres"
    _depends_are_healthy(
        app,
        ("postgres", "milvus", "planner-llm", "generation-llm", "embedding", "reranker", "nli"),
    )


def test_streamlit_static_structure_has_distinct_host_ports(services):
    expected_ports = {
        "streamlit-api": ["${API_CPU_STREAMLIT_PORT:-8501}:8501"],
        "streamlit-gpu": ["${GPU_STREAMLIT_PORT:-8502}:8501"],
    }
    assert services["streamlit-api"]["environment"]["STREAMLIT_PORT"] == "8501"
    assert services["streamlit-gpu"]["environment"]["STREAMLIT_PORT"] == "8501"
    for service_name, ports in expected_ports.items():
        assert services[service_name]["ports"] == ports
    default_host_ports = [ports[0].split(":-")[1].split("}")[0] for ports in expected_ports.values()]
    assert len(default_host_ports) == len(set(default_host_ports))


def test_streamlit_static_structure_shares_common_anchor_fields(compose, services):
    # PyYAML validates the merged static shape, not Docker Compose interpolation semantics.
    common = compose["x-streamlit-common"]
    assert "ports" not in common
    assert COMMON_FIELDS <= common.keys()
    for field in COMMON_FIELDS:
        assert services["streamlit-api"][field] == common[field]
        assert services["streamlit-gpu"][field] == common[field]
    assert "urllib" in _healthcheck_url(common)
    assert "/_stcore/health" in _healthcheck_url(common)
    assert "alembic upgrade head" in common["command"]
    assert "streamlit run app.py" in common["command"]


def test_static_internal_urls_resolve_to_declared_compose_services(services):
    for app_name in ("streamlit-api", "streamlit-gpu"):
        for key, value in services[app_name]["environment"].items():
            if key.endswith("_SERVICE_URL") or key.endswith("_LLM_BASE_URL"):
                if not value.startswith("http://"):
                    continue
                assert urlparse(value).hostname in services


def test_static_cpu_inference_is_offline_ready_and_loopback_bound(services):
    for name in ("embedding-cpu", "nli-cpu", "reranker-cpu"):
        service = services[name]
        assert service["environment"]["HF_HUB_OFFLINE"] == "${HF_HUB_OFFLINE:-0}"
        assert service["environment"]["TRANSFORMERS_OFFLINE"] == "${TRANSFORMERS_OFFLINE:-0}"
        assert "/ready" in _healthcheck_url(service)
        assert all(port.startswith("127.0.0.1") for port in service["ports"])


def test_static_inference_healthchecks_gate_on_readiness(services):
    for name in ("embedding-cpu", "nli-cpu", "embedding", "reranker", "nli"):
        healthcheck = services[name]["healthcheck"]
        assert "/ready" in _healthcheck_url(services[name])
        assert healthcheck["start_period"] in {"120s", "180s", "240s", "300s"}
    if "reranker-cpu" in services:
        assert "/ready" in _healthcheck_url(services["reranker-cpu"])


def test_static_vllm_healthchecks_and_healthy_app_dependencies(services):
    for name in ("planner-llm", "generation-llm"):
        assert "/health" in _healthcheck_url(services[name])
        assert services["streamlit-gpu"]["depends_on"][name]["condition"] == "service_healthy"


def test_static_compose_contains_no_hardcoded_real_api_keys(compose):
    serialized = yaml.safe_dump(compose)
    assert "sk-" not in serialized
    for app_name in ("streamlit-api", "streamlit-gpu"):
        deploy_config = compose["services"][app_name].get("deploy", {})
        assert "device_ids" not in yaml.safe_dump(deploy_config)


def test_minio_static_credentials_remain_milvus_default_cloud_blocker(services):
    # Deliberate risk state: changing these defaults requires synchronizing Milvus credentials.
    assert services["minio"]["environment"] == {
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
    }
    milvus_env = services["milvus"]["environment"]
    assert "MINIO_ACCESS_KEY_ID" not in milvus_env
    assert "MINIO_SECRET_ACCESS_KEY" not in milvus_env
    assert "${" not in yaml.safe_dump(services["minio"]["environment"])


def test_static_milvus_and_attu_use_healthy_dependency_gates(services):
    for app_name in ("streamlit-api", "streamlit-gpu"):
        assert services[app_name]["depends_on"]["milvus"]["condition"] == "service_healthy"
    assert services["attu"]["depends_on"]["milvus"]["condition"] == "service_healthy"
