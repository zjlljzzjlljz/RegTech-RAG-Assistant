# Production Upgrade

## Runtime topology

- `planner-llm`: vLLM with Qwen2.5-7B-Instruct.
- `generation-llm`: vLLM with Qwen2.5-72B-Instruct-AWQ for drafting, auditing, and evaluation judging.
- `embedding`: BGE-M3 dense and native lexical sparse weights.
- `reranker`: BGE reranker batch inference.
- `nli`: multilingual entailment verification for prose grounding.
- `milvus`, `etcd`, `minio`: versioned hybrid retrieval storage.
- `postgres`: append-oriented audit records managed by Alembic.

Start the production-like stack with:

```bash
docker compose --profile app --profile gpu up -d
```

The 72B AWQ service is configured for two GPUs. Adjust tensor parallelism and GPU reservations to the deployed hardware.

## Index migration

The default collection is `regtech_compliance_chunks_v2`. It is intentionally separate from the legacy blake2b collection because learned sparse token IDs are not compatible with legacy sparse vectors.

```bash
python -m src.indexing.rebuild_milvus \
  --pdf-dir data/raw_pdfs \
  --drop-existing
```

Ingest and query must use the same BGE-M3 model image and revision. Production should set `PREFER_LOCAL_INFERENCE_FALLBACK=false`.

## Safety behavior

Auditor parse errors, missing approval fields, model failures, and NLI failures are fail-closed. A rejected draft that reaches the iteration limit is not returned; the user sees `本报告未经审计通过`, while the internal audit record retains status and evidence metadata.

## Evaluation

`--suite all` combines the existing English, Chinese, and HK-mixed query files into a 75-query retrieval suite.

```bash
python -m src.evaluation.eval_retrieval \
  --suite all --fusion rrf --with-rerank \
  --output reports/candidate_metrics.json

python -m src.evaluation.regression_gate \
  --baseline src/evaluation/baseline_metrics.json \
  --candidate reports/candidate_metrics.json
```

The checked-in baseline is a schema placeholder. Replace it with metrics produced from the rebuilt v2 collection before enabling the evaluation job as a required branch check.
