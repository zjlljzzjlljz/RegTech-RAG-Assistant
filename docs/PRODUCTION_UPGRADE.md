# Production Upgrade

> 文档状态：生产升级与验证清单
> 最近核对：2026-07-15
> 当前架构：[`ARCHITECTURE.md`](ARCHITECTURE.md)
> 开发规范：[`../PROJECT_RULES.md`](../PROJECT_RULES.md)

本文区分“已实现”“待验证”和“生产目标”。服务已写入代码或 Docker Compose，不代表已在目标硬件完成生产验收。

## 1. Status Summary

| 能力 | 状态 | 说明 |
|---|---|---|
| Milvus v2 hybrid collection | 已实现/已验证检索 | `regtech_compliance_chunks_v2`，BGE-M3 dense + sparse |
| Semantic parent-child chunking | 已实现/已测试 | 1500/400 token，200 token overlap，stable ID |
| Parent backfill | 已实现/已进入基线 | 子块命中后按 `parent_id` 回填 |
| OpenAI-compatible LLM client | 已实现 | 支持 DeepSeek、vLLM、TGI-compatible endpoint |
| DeepSeek runtime | 已实现/外部 API | 当前低成本运行选项，不满足本地数据驻留目标 |
| Planner vLLM 7B | 生产目标/待验证 | Qwen2.5-7B-Instruct，单 GPU 配置 |
| Generation vLLM 72B AWQ | 生产目标/待验证 | Qwen2.5-72B-Instruct-AWQ，双 GPU 配置 |
| BGE-M3 inference service | 已实现/部分验证 | CPU 服务可用；GPU 服务待目标硬件验证 |
| BGE reranker service | 已实现/待验证 | 本机 CPU 单查询超过 8 分钟，未完成 75 条评估 |
| Conditional HyDE | 已实现/待验证 | 无有效 reranker score 时保持禁用 |
| NLI service | 已实现/待扩充评估 | fail-closed，需更多多语言标注校准 |
| PostgreSQL audit repository | 已实现 | Compose 与 Alembic 已具备；本地可回退 SQLite |
| CI unit tests | 已实现 | compile check + `pytest -q` |
| CI retrieval regression gate | 已实现/条件执行 | candidate 文件存在时执行，尚非 required check |

## 2. Runtime Topology

### 2.1 已实现的 Compose 服务

- `postgres`: PostgreSQL 16 审计存储。
- `migrate`: Alembic migration 工具容器。
- `milvus`, `etcd`, `minio`: Milvus standalone 依赖与持久化。
- `attu`: Milvus 管理界面。
- `streamlit`: 应用容器，启动前执行 Alembic upgrade。
- `embedding-cpu`: BGE-M3 CPU 推理服务。
- `reranker-cpu`: BGE-Reranker-Large CPU 服务，仅适合功能验证或离线实验。

### 2.2 生产目标 GPU 服务

- `planner-llm`: vLLM + Qwen2.5-7B-Instruct。
- `generation-llm`: vLLM + Qwen2.5-72B-Instruct-AWQ，用于 draft、audit 和 judge。
- `embedding`: BGE-M3 dense 与 native lexical sparse weights。
- `reranker`: BGE-Reranker-Large batch inference。
- `nli`: multilingual entailment verification。

生产式目标启动命令：

```bash
docker compose --profile app --profile gpu up -d
```

该命令需要 NVIDIA Container Toolkit 和足够 GPU 资源。72B AWQ 服务当前配置 `tensor-parallel-size=2`，部署前必须按实际 GPU 显存、模型上下文和并发目标重新核算。

### 2.3 本机可执行范围

当前本机 8 GB 内存不适合运行 70B/72B 4-bit 模型。可以运行基础设施、应用及部分 CPU 推理服务，但 BGE-Reranker-Large 的 CPU 延迟不满足交互式 SLO。

仅启动基础设施：

```bash
docker compose up -d postgres etcd minio milvus attu
```

启动 CPU embedding 服务：

```bash
docker compose --profile cpu up -d embedding-cpu
```

除离线验证外，不建议在本机启动 `reranker-cpu` 执行完整 75 条评估。

## 3. LLM Migration

### 3.1 已实现

角色模型通过以下环境变量独立配置：

```text
PLANNER_LLM_PROVIDER / MODEL / BASE_URL / API_KEY
DRAFT_LLM_PROVIDER   / MODEL / BASE_URL / API_KEY
AUDITOR_LLM_PROVIDER / MODEL / BASE_URL / API_KEY
JUDGE_LLM_PROVIDER   / MODEL / BASE_URL / API_KEY
```

OpenAI-compatible client 允许 DeepSeek 和本地 vLLM 使用同一业务接口。切换模型时不得修改查询规划、检索或 Agent 业务代码。

### 3.2 当前外部 API 运行方式

DeepSeek 可以作为当前低成本 provider。API key 只允许存放在 `.env` 或 secret manager，不得进入 Git、日志或文档。

使用外部 API 时，不得发送未经批准的客户身份、交易数据或内部监管材料。外部 API 配置不满足本地数据驻留目标。

### 3.3 生产目标

```text
planner  -> Qwen2.5-7B-Instruct on vLLM
draft    -> Qwen2.5-72B-Instruct-AWQ on vLLM
auditor  -> Qwen2.5-72B-Instruct-AWQ on vLLM
judge    -> Qwen2.5-72B-Instruct-AWQ on vLLM
```

切换前需要完成：

1. OpenAI-compatible API contract test。
2. JSON mode 和 fail-closed auditor 测试。
3. 中英文/HK-mixed 输出质量评估。
4. p50/p95 延迟、吞吐、GPU 内存和超时测试。
5. Prompt cache 隔离与数据保留检查。

## 4. Index Migration

### 4.1 已实现

默认 collection：

```text
regtech_compliance_chunks_v2
```

它与 legacy blake2b collection 分离，因为 BGE-M3 学习型 sparse token ID 与历史 sparse vector 不兼容。

重建命令：

```bash
python -m src.indexing.rebuild_milvus \
  --pdf-dir data/raw_pdfs \
  --drop-existing
```

### 4.2 强制一致性要求

入库和查询必须使用相同：

- BGE-M3 model name 与 revision。
- tokenizer 与文本预处理。
- dense normalization。
- native sparse weight 生成方式。
- semantic chunk version 与 token 参数。

生产环境应设置：

```text
PREFER_LOCAL_INFERENCE_FALLBACK=false
```

这样远程推理服务失败时不会静默切换到未固定版本的本地模型。

### 4.3 待完善

每次重建应生成 index manifest，至少记录：

- collection 和 schema version。
- 原始语料文件哈希。
- BGE-M3 image/model revision。
- parent/child/overlap 配置。
- chunk count、构建时间和构建代码 commit。

新 collection 在完成 75 条回归评估前不得切换默认流量。

## 5. Retrieval Validation

### 5.1 已验证基线

基线文件：`src/evaluation/baseline_metrics.json`

| 指标 | 当前值 |
|---|---:|
| Hit@1 | 0.5467 |
| Hit@3 | 0.7600 |
| Hit@5 | 0.8667 |
| Recall@10 | 0.8933 |
| MRR | 0.6734 |

基线包含 75 条 EN/ZH/HK-mixed 查询，模式为 conditional multi-query + weighted RRF + parent backfill，`reranked=false`。

### 5.2 Weighted RRF 结论

- 直接检索的最佳扫描点为 dense:sparse `8:1`，Recall@10 `0.88`、MRR `0.6247`。
- 完整规划链路在 `8:1` 下为 Recall@10 `0.88`、MRR `0.6106`。
- 两者没有超过当前基线，因此默认 dense:sparse 仍为 `1:1`，原查询权重为 `2.0`。

### 5.3 待验证：Reranker 与 HyDE

BGE-Reranker-Large 在本机 CPU 上单查询超过 8 分钟，因此尚未完成：

1. 75 条 reranked retrieval 指标。
2. 有/无 reranker 消融。
3. 有/无条件 HyDE 消融。
4. HyDE 阈值与 score margin 校准。
5. 全链路 p50/p95 延迟和成本比较。

只有 GPU 全量结果证明质量收益且延迟可接受后，才能把 reranker 和 HyDE 标为“已验证”。

## 6. Safety Behavior

### 6.1 已实现

- Auditor 非法 JSON 时 fail-closed。
- `approved` 缺失或不是 boolean 时 fail-closed。
- 模型调用、检索或 NLI 错误进入 error 状态。
- rejected draft 最多修订三轮。
- 达到最大轮次后不返回草稿，用户看到 `本报告未经审计通过`。
- 结构化 claims 必须引用实际检索 chunk。
- 散文正文经过多语言 NLI grounding。
- 内部审计记录保留状态、证据和错误元数据。

### 6.2 待验证

- 扩大 NLI 中英文/HK-mixed contradiction 数据集。
- 校准 `NLI_ENTAILMENT_THRESHOLD`。
- 对 provider timeout、断连、无效 JSON 和半响应执行故障注入。
- 验证 PostgreSQL 在失败路径上的审计记录完整性。
- 由合规人员人工抽检被批准与被阻断样例。

## 7. PostgreSQL Migration

### 7.1 已实现

- PostgreSQL 16 Compose service。
- `PostgresTransactionRepository`。
- Alembic 配置和 transaction log migration。
- `DATABASE_URL` 存在时自动选择 PostgreSQL。
- Streamlit 容器启动前执行 `alembic upgrade head`。

### 7.2 当前兼容行为

未设置 `DATABASE_URL` 时，应用回退到 SQLite `TransactionRepository`。此行为便于本地开发，但生产环境不得依赖该回退。

### 7.3 生产验收项

1. migration upgrade 验证。
2. 备份与恢复演练。
3. 连接池、并发写入和重启测试。
4. append-oriented 审计记录不可变性检查。
5. 数据保留、脱敏和访问控制策略。
6. PostgreSQL 不可用时的 fail-closed 与告警行为。

## 8. Evaluation And CI

### 8.1 检索评估

`--suite all` 组合英文、中文和 HK-mixed 查询文件：

```bash
python -m src.evaluation.eval_retrieval \
  --suite all --fusion rrf \
  --output reports/candidate_metrics.json
```

完成 GPU reranker 部署后再运行：

```bash
python -m src.evaluation.eval_retrieval \
  --suite all --fusion rrf --with-rerank \
  --output reports/candidate_reranked_metrics.json
```

回归比较：

```bash
python -m src.evaluation.regression_gate \
  --baseline src/evaluation/baseline_metrics.json \
  --candidate reports/candidate_metrics.json
```

### 8.2 当前 CI 状态

`.github/workflows/ci.yml` 已运行：

- Python compile check。
- `pytest -q`。
- candidate metrics 存在时执行 regression gate。

评估 job 当前为条件执行。启用 required branch check 前，应确保 CI 能访问固定 Milvus snapshot 或使用可复现的评估环境，并避免在普通 PR 中意外调用付费 LLM API。

### 8.3 生产目标

- PR：确定性单元/集成测试与安全路径门禁。
- Nightly/manual：完整检索、reranker、HyDE 和生成评估。
- Release：固定 collection、模型 revision、评估集和阈值的验收报告。
- 任一 fail-closed 测试失败时阻断发布。

## 9. Production Readiness Checklist

在把系统描述为生产就绪前，必须全部完成：

- [ ] 目标 GPU 上成功启动 planner、generation、embedding、reranker 和 NLI 服务。
- [ ] 固定所有模型 image 与 revision。
- [ ] 完成 75 条 reranker + conditional HyDE 全量消融。
- [ ] 满足约定的 p95 延迟和错误率。
- [ ] PostgreSQL migration、备份、恢复和并发写入通过。
- [ ] `PREFER_LOCAL_INFERENCE_FALLBACK=false` 下完成故障测试。
- [ ] NLI 阈值经过多语言标注集校准。
- [ ] required CI regression gate 启用。
- [ ] API key、日志、数据保留与访问控制通过安全审查。
- [ ] 人工合规抽检完成并记录结论。

未完成上述清单前，项目应描述为“生产化原型”或“production-like architecture”，不得描述为已投入监管生产环境。
