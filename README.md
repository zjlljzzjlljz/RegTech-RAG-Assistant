# RegTech RAG Assistant — Auditable RAG for HKMA AML/CFT Compliance

A prototype regulatory compliance assistant that retrieves, drafts, audits, and grounds answers using only official HKMA regulatory materials. Built on hybrid retrieval (Milvus + BGE-M3), a LangGraph Draft-Audit feedback loop, and NLI grounding verification.

> ⚠️ **Prototype status.** This is a research and demonstration system. It is not a legal advice tool, regulatory decision engine, or production compliance platform.

---

## What It Does

1. **Ingest & index** HKMA AML/CFT PDF documents with semantic parent-child chunking and stable IDs.
2. **Hybrid retrieval** — BGE-M3 dense + learned sparse embeddings, RRF fusion, query expansion, parent-context backfill.
3. **Agentic Draft-Audit loop** — LangGraph state machine generates a compliance answer, audits it against source evidence, and revises up to 3 iterations. Drafts that fail audit are blocked with an explicit watermark.
4. **NLI grounding** — Multi-lingual NLI (mDeBERTa) cross-checks generated prose against retrieved sources.
5. **Audit trail** — PostgreSQL + Alembic for append-only audit records.

---

## Current Baseline

| Metric | Value |
|---|---|
| Hit@1 | 0.5467 |
| Hit@3 | 0.7600 |
| Hit@5 | 0.8667 |
| Recall@10 | 0.8933 |
| MRR | 0.6734 |

*Evaluation set: 25 regulatory questions × 3 language variants (EN/ZH/HK-mixed) = 75 retrieval records. Last test run: 41 passed.*

---

## Tech Stack

| Layer | Implementation |
|---|---|
| Orchestration | LangGraph (Draft-Audit state machine) |
| Embeddings | BGE-M3 (dense + native lexical sparse) |
| Vector DB | Milvus (collection `regtech_compliance_chunks_v2`) |
| LLM | OpenAI-compatible provider abstraction; currently configured for DeepSeek API |
| NLI Grounding | mDeBERTa-based prose + claims verification |
| Audit Storage | PostgreSQL + Alembic |
| Deployment | Docker Compose (app, Milvus, PostgreSQL, optional GPU inference) |
| Testing | pytest |
| Evaluation | Custom retrieval suite + RAGAS integration |

---

## Quick Start

```bash
# Clone, install, and create the local configuration file
git clone https://github.com/zjlljzzjlljz/RegTech-RAG-Assistant.git
cd RegTech-RAG-Assistant
pip install -r requirements.txt
cp .env.example .env
```

Before starting API+CPU mode, set `LLM_API_KEY` in the untracked `.env`, process environment, or a secret manager. `.env.example` intentionally leaves it empty; Compose maps an empty value, but `Settings` rejects API+CPU startup until a key is injected. GPU self-hosted mode does not require a DeepSeek key. Keep real keys out of Git. Role-specific variables such as `PLANNER_LLM_MODEL` override the global LLM configuration; unspecified role settings inherit the corresponding global provider, base URL, and API key.

Configuration precedence is: process environment, Compose/cloud/secret injection > host `.env` > code defaults. In host mode, Python loads `.env` with `override=False`. In container modes Python does not read `/app/.env`; Compose can still interpolate the host environment or repository `.env`, and injects only variables explicitly listed in each service definition.

### API LLM + CPU inference containers

```bash
docker compose --profile api-cpu up -d
```

Open Streamlit at <http://localhost:8501> (override the host port with `API_CPU_STREAMLIT_PORT`; the container still listens on `8501`). This profile accepts only role endpoints using HTTPS, the exact hostname `api.deepseek.com`, port `443` or the implicit HTTPS port, and no URL credentials. It sends role LLM requests to the configured external OpenAI-compatible API and loads BGE-M3 embedding plus multilingual NLI models on CPU. Initial model download/loading can be slow and memory-intensive; CPU performance has not been dynamically validated. The default retrieval path is **hybrid dense+sparse RRF without reranking**. The optional CPU reranker is an experimental `tools`-profile service on host port `8102`, not part of this default.

External API data boundary: prompts and retrieved context sent to DeepSeek leave the local deployment. Do not send customer identity, transaction data, confidential internal material, or other data that has not been approved for that provider.

### Self-hosted GPU target

```bash
docker compose --profile gpu-self-hosted up -d
```

Open Streamlit at <http://localhost:8502> (override the host port with `GPU_STREAMLIT_PORT`; the container still listens on `8501`). This mode uses internal vLLM services and does not consume `LLM_API_KEY`. This is a target topology and has not been dynamically validated. `generation-llm` uses tensor parallel size 2 and must see two GPUs at the same time. Planner, embedding, reranker, and NLI services also request GPUs. Compose declares GPU counts but no device IDs or explicit cross-service isolation, so services may contend for GPU memory. Allocate devices and capacity for the actual hardware before use; the file does not establish a fixed total GPU requirement.

### Host application + container infrastructure

```bash
docker compose --profile api-cpu up -d \
  postgres etcd minio milvus embedding-cpu nli-cpu
streamlit run app.py
```

Open Streamlit at <http://localhost:8501>. The host application connects to PostgreSQL at `127.0.0.1:5432`, Milvus at `127.0.0.1:19530`, embedding at `localhost:8101`, and NLI at `localhost:8103`. `NLI_ENABLED=true`, `PREFER_LOCAL_INFERENCE_FALLBACK=false`, and reranking is disabled by default in `.env.example`.

Compose profiles are additive, not mutually exclusive. `tools` starts Attu and the experimental CPU reranker when explicitly selected:

```bash
docker compose --profile tools up -d attu reranker-cpu
```

PostgreSQL, Milvus, MinIO, CPU inference, and Attu host publications are loopback-only. The Streamlit ports are configurable but bind to the host interfaces selected by Docker; Streamlit has no built-in authentication. Production deployments must place it behind an authenticating reverse proxy or API gateway and restrict ingress with cloud firewall/security-group rules.

Inference `/ready` uses one shared single-flight model-loading path: concurrent readiness and business requests observe the same loading state, so business endpoints do not start duplicate model loads. `/health` remains lightweight liveness.

MinIO deliberately retains the Milvus standalone-compatible `minioadmin/minioadmin` default in this change to avoid breaking synchronized service access. This is a cloud-deployment blocker, so API+CPU must not be described as safely production-ready. Before cloud deployment, change the MinIO root credentials and Milvus `MINIO_ACCESS_KEY_ID`/`MINIO_SECRET_ACCESS_KEY` together, inject both through a secret manager, and verify access to existing data. This change does not claim to resolve that credential work.

For a static Compose check that does not print expanded configuration, use a dummy API key:

```bash
LLM_API_KEY=dummy docker compose --profile api-cpu config --quiet
LLM_API_KEY=dummy docker compose --profile gpu-self-hosted config --quiet
```

This documentation change introduces no migration, Milvus collection change, or Docker volume change. Starting a Streamlit application container still runs the existing `alembic upgrade head` command, so operators must follow the project’s existing database-upgrade procedure.

---

## Architecture

```
User Query
   → Query Planner (intent classification + multi-query expansion)
   → Hybrid Retrieval (BGE-M3 dense + sparse → RRF fusion → optional reranker)
   → Parent-Context Backfill
   → Draft-Audit-Revise Loop (LangGraph, max 3 iterations)
   → NLI Grounding Verification
   → Finalize (approved / blocked / insufficient-evidence)
   → Audit Trail (PostgreSQL + Alembic)
```

### Safety Constraints

- **Fail-closed** — auditor errors, JSON parse failures, timeouts, or max iterations all trigger rejection. Nothing crosses the user boundary without audit approval.
- **Source-locked** — every factual claim must trace to a specific chunk ID from retrieved evidence. Claims without valid source IDs are stripped.
- **Insufficient evidence** — queries the corpus cannot answer are explicitly refused, not hallucinated.
- **Watermarked** — blocked drafts display `本报告未经审计通过` and are never returned as approved content.

---

## Known Limitations (Not Yet Validated)

- **BGE-Reranker-Large** — currently 8+ minutes per query on CPU. Full 75-query end-to-end validation has not been completed. GPU inference service is the target topology.
- **Conditional HyDE** — query expansion via HyDE is gated behind reranker scores. Since reranker validation is incomplete, HyDE remains disabled for all practical paths.
- **72B AWQ model** — the target LLM (Qwen2.5-72B-Instruct-AWQ) requires dual-GPU inference that is not available on the current development machine.
- **DeepSeek API** — currently used as the external LLM provider. This does not satisfy HKMA data-residency requirements for production deployment.
- **Evaluation set size** — 75 records is adequate for development gating but insufficient for production calibration. A separate held-out validation set is needed.

---

## Repository Structure

```
.
├── app.py                     # Application entry point
├── config/                    # Settings, model roles, collection config
├── src/
│   ├── agent/                 # LangGraph Draft-Audit state machine
│   ├── indexing/              # Semantic chunker, Milvus ingestion, sparse tokenizer
│   ├── inference/             # Provider-neutral LLM abstraction
│   ├── retrieval/             # Query pipeline (HyDE, RRF, reranker, parent backfill)
│   ├── safety/                # NLI grounding verifier
│   ├── storage/               # PostgreSQL repositories
│   └── evaluation/            # Retrieval and generation evaluation
├── services/                  # Inference API service definitions
├── migrations/                # Alembic schema migrations
├── tests/                     # pytest suite
├── data/                      # Regulatory PDFs (not committed)
├── docker-compose.yml
├── Makefile
├── PROJECT_RULES.md           # Full governance & development rules (SSOT)
├── docs/                      # Architecture docs, ADRs, production upgrade plan
└── archive/                   # Historical prototype artifacts (ChromaDB, MiniLM, Claude)
```

---

## Development

This project follows strict module-based development governed by `PROJECT_RULES.md` (Single Source of Truth). Key constraints:

- **One module at a time** — no parallel development.
- **Fail-closed by default** — safety critical paths must never degrade to open.
- **Architecture review before code** — every new module requires explicit design approval.
- **No unvalidated claims** — performance, accuracy, or production readiness claims must be backed by data.

---

## License

This project is for academic and demonstration purposes. HKMA regulatory materials are publicly available and remain the property of the Hong Kong Monetary Authority.
