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
# Clone and install
git clone https://github.com/zjlljzzjlljz/RegTech-RAG-Assistant.git
cd RegTech-RAG-Assistant
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and endpoints

# Start infrastructure
docker compose --profile app up -d

# Index documents (requires HKMA PDFs in data/ directory)
python -m src.indexing.milvus_ingest

# Run the system
python app.py
```

---

## Architecture

```
User Query
   → Query Planner (intent classification + multi-query expansion)
   → Hybrid Retrieval (BGE-M3 dense + sparse → RRF fusion → reranker)
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
