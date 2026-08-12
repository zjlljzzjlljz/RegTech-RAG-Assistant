# ADR 0001: Separate API+CPU and GPU Deployment Topologies

- Status: Accepted
- Date: 2026-07-26
- Decision maker: Project owner
- Approval basis: The project owner explicitly requested correction of this accepted decision on 2026-07-26.

## Context

The application supports host, external-API plus CPU inference, and self-hosted GPU modes. They have different provider, secret, data-boundary, service, port, and hardware requirements. Compose profiles are additive and are not mutually exclusive.

## Decision

Retain `host` as the default Python deployment mode and define separate `api-cpu` and `gpu-self-hosted` Compose profiles. Keep two explicit Streamlit services derived from a shared `x-streamlit-common` anchor. The anchor contains common application behavior but no host port; each service declares its own mapping.

### API+CPU topology

- `streamlit-api` accepts DeepSeek role URLs only when they use HTTPS, the exact hostname `api.deepseek.com`, port `443` or the implicit HTTPS port, and no URL credentials.
- Compose maps `LLM_API_KEY` with an empty default, avoiding cross-profile interpolation failure. `Settings` still rejects API+CPU startup unless every role resolves a non-empty key.
- Real keys belong only in an untracked `.env`, process environment, or secret manager.
- `embedding-cpu` and `nli-cpu` are required; reranking remains disabled by default. CPU NLI performance has not been dynamically validated.
- Prompts and retrieved context sent to DeepSeek cross the local data boundary and require explicit approval.

### GPU target topology

- `streamlit-gpu` uses internal planner/generation vLLM, embedding, reranker, and NLI services; it does not require or receive a DeepSeek key.
- `generation-llm` requires simultaneous visibility of two GPUs because `tensor-parallel-size=2`; other services also request GPUs. Device placement, total capacity, and end-to-end operation have not been dynamically validated.

### Network and operator tooling

- API+CPU defaults to host port `8501`; GPU defaults to `8502`. Both containers listen on `8501`. Operators may override host ports with `API_CPU_STREAMLIT_PORT` and `GPU_STREAMLIT_PORT`.
- PostgreSQL, Milvus, MinIO, CPU inference, and Attu publish to loopback.
- Attu and experimental `reranker-cpu` belong only to the `tools` profile.
- Bare Streamlit provides no authentication. Production requires an authenticating reverse proxy or API gateway plus cloud firewall/security-group ingress controls.

### Readiness

Inference `/ready` and business endpoints share a single-flight model-loading state machine. Concurrent requests observe one loading attempt; business endpoints do not trigger duplicate loads. `/health` remains lightweight liveness.

### MinIO credential blocker

The official Milvus standalone-compatible `minioadmin/minioadmin` pairing remains unchanged in this decision to avoid breaking synchronization between MinIO and Milvus or access to existing data. This is a cloud-deployment blocker, and API+CPU is not safely production-ready while it remains. Before cloud deployment, operators must update MinIO root credentials and Milvus `MINIO_ACCESS_KEY_ID`/`MINIO_SECRET_ACCESS_KEY` together, inject them through a secret manager, and verify existing-data access. This decision does not claim that credential work is complete.

## Configuration loading policy

Precedence is process environment, Compose/cloud/secret injection > host `.env` > code defaults. Host mode loads `.env` with `override=False`; container Python does not read `/app/.env`. Compose injects only explicitly listed service variables. Global LLM values may be inherited, while role-prefixed values override them.

## Consequences

- Startup commands select profiles explicitly, but selected profiles can run together.
- Container connections use Compose DNS; host-accessible infrastructure and CPU inference use loopback publications.
- External API use requires secret handling and an approved data-boundary decision.
- CPU NLI performance and the GPU target remain unverified.
- Production exposure requires authentication and network controls beyond Streamlit.

## Alternatives considered

Compose override files and one parameterized Streamlit service were considered. Profiles keep the current single-file development contract while explicit services preserve topology-specific validation and dependencies.

## Data and migration impact

This decision does not change database migrations, Milvus collections, persisted data, or Docker volumes.

## Rollback

Stop the selected profile and return to host mode or a previous Compose invocation. No data rollback, collection rebuild, migration downgrade, or volume conversion is required.
