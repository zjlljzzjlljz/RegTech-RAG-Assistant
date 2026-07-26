# Development Log

## 2026-07-26

- Synchronized `.env.example` with empty-key handling and configurable API/GPU Streamlit host ports.
- Added the direct Python 3.11-compatible dependency `PyYAML==6.0`, matching the current environment.
- Corrected README, architecture, production-upgrade guidance, and accepted ADR 0001 to match implemented API+CPU and GPU topology contracts.
- Documented strict DeepSeek HTTPS endpoint validation, GPU key independence, additive profiles, loopback publications, Attu/tools scope, and unauthenticated Streamlit production controls.
- Documented the shared single-flight inference readiness path and preserved the unverified status of CPU NLI performance and the GPU target.
- Recorded the external API data boundary and the unchanged `minioadmin/minioadmin` pairing as a cloud-deployment blocker requiring synchronized secret-managed rotation and existing-data verification. This work does not claim to resolve MinIO credentials.
- Confirmed the scope does not change migrations, Milvus collections, persisted data, or Docker volumes.

### Verification status

- Docker Desktop 4.81.0, Docker Engine 29.6.1, Docker Compose v5.2.0, context `desktop-linux`.
- All Docker commands used the absolute Docker executable path with `--env-file /dev/null`. Validation ran only `compose config`; no containers were started, built, or pulled.
- API config, GPU config without keys, base config without profiles, tools config, and the combined-profile config all completed with `--quiet` and exit 0.
- Resolved service sets:
  - API: `[etcd, minio, milvus, nli-cpu, postgres, embedding-cpu, streamlit-api]`
  - GPU: `[etcd, minio, milvus, reranker, embedding, generation-llm, nli, planner-llm, postgres, streamlit-gpu]`
  - Base: `[etcd, minio, milvus, postgres]`
  - Tools: `[postgres, reranker-cpu, etcd, minio, milvus, attu, migrate]`
- Streamlit port mappings resolve to `streamlit-api` 8501→8501 and `streamlit-gpu` 8502→8501. All loopback-focused port publications match the design.
- `python -m pytest -q tests/config/test_settings.py`: 61 passed in 0.88s.
- `python -m pytest -q tests/deployment/test_compose_contract.py`: 12 passed in 0.60s.
- `python -m pytest -q tests/services/test_inference_api_readiness.py`: 9 passed in 1.13s.
- `python -m pytest -q`: 123 passed, 6 warnings in 5.39s.
- `compileall` for the specified directories: exit 0.
- `git diff --check`: exit 0.
- Containers, models, end-to-end question answering, APIs, and RAGAS were not run. These results do not claim dynamic deployment validation or cloud readiness.
- The unchanged `minioadmin/minioadmin` credentials remain a cloud-deployment blocker requiring synchronized secret-managed rotation and existing-data verification.
- Streamlit remains unauthenticated; production exposure requires authentication and the documented production controls.
