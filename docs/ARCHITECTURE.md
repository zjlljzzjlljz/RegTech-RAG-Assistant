# RegTech RAG Assistant - 当前工程架构

> 文档状态：当前架构说明
> 最近核对：2026-07-15
> 开发规范：以根目录 [`PROJECT_RULES.md`](../PROJECT_RULES.md) 为唯一权威来源
> 历史原型：见 [`superpowers/specs/ARCHITECTURE_LEGACY.md`](superpowers/specs/ARCHITECTURE_LEGACY.md)

本文只描述当前 Milvus、BGE-M3、OpenAI-compatible LLM client、LangGraph 和 PostgreSQL 架构。ChromaDB、MiniLM、早期 Claude 单一接入及编号脚本属于历史原型，不再作为当前实现依据。

## 1. 状态定义

为避免把“代码存在”误写成“生产已验证”，本文使用以下状态：

| 状态 | 含义 |
|---|---|
| 已实现 | 代码、配置或部署定义已经存在 |
| 已验证 | 已在当前环境完成相应测试或评估 |
| 待验证 | 已实现，但受硬件、费用或完整评估限制，尚未端到端验收 |
| 生产目标 | 计划用于生产式部署，不能据此宣称已经上线 |

## 2. 项目定位

本项目是面向香港银行 AML/CFT 合规场景的 RegTech RAG 助手。系统从 HKMA 公开监管材料中检索证据，为英文、中文及香港混合表达查询生成可追溯回答，并通过审计循环、claim 校验和 NLI 据实性验证降低无依据陈述进入用户结果的风险。

系统是合规研究辅助工具，不替代法律意见、监管裁决或合规人员的最终判断。

### 2.1 当前核心能力

- Milvus 稠密与学习型稀疏双路检索。
- BGE-M3 同时生成 dense embedding 与 native lexical sparse weights。
- 语义父子切块、200 token overlap、稳定 `chunk_id` 与 `parent_id`。
- 多语言查询扩展、加权 RRF、可选 Cross-Encoder 重排。
- 仅在有效重排分数不足时触发的条件 HyDE。
- 查询规划缓存与并行检索。
- LangGraph 起草、审查、修订闭环，最多三轮。
- 审查解析错误和模型失败 fail-closed。
- 最大轮次阻断及 `本报告未经审计通过` 水印。
- claims/source 映射与散文正文 NLI grounding。
- PostgreSQL 审计日志及 Alembic migration；本地仍保留 SQLite 回退。
- Docker Compose 基础设施、CPU/GPU 推理服务定义与 CI 回归门禁。

## 3. 架构全景

```text
                         Streamlit / app.py
                                  |
                                  v
                     ComplianceAgentGraph
                                  |
             +--------------------+--------------------+
             |                                         |
             v                                         v
 ComplianceRetrievalPipeline                 Draft / Auditor LLM
             |                                  role-based clients
             |                                         |
    +--------+---------+                               v
    |                  |                     OpenAI-compatible API
    v                  v                    DeepSeek or local vLLM
BGE-M3 encoder      Query planner
dense + sparse      intent / multi-query / HyDE
    |                  |
    +--------+---------+
             |
             v
       Milvus hybrid search
       dense + sparse arms
             |
             v
       weighted RRF fusion
             |
             v
   optional BGE reranker service
             |
             v
       parent context backfill
             |
             v
    draft -> audit -> NLI grounding
             |
      approved / blocked / error
             |
             v
 PostgreSQL audit repository
 (SQLite fallback for local use)
```

## 4. 核心模块与依赖

| 模块 | 主要文件 | 当前职责 | 状态 |
|---|---|---|---|
| 配置 | `config/settings.py` | 模型角色、Milvus、切块、检索、推理和存储配置 | 已实现 |
| LLM client | `src/inference/llm_client.py` | OpenAI-compatible 与兼容 adapter、统一响应和 JSON 解析 | 已实现 |
| 语义切块 | `src/indexing/semantic_chunker.py` | 句段边界、父子窗口、overlap、稳定 ID | 已实现/已测试 |
| BGE-M3 编码 | `src/indexing/milvus_ingest.py` | dense 与 native sparse 一致编码 | 已实现/已验证检索 |
| Milvus 存储 | `src/indexing/milvus_ingest.py` | collection schema、写入、dense/sparse 查询 | 已实现/已验证检索 |
| 索引重建 | `src/indexing/rebuild_milvus.py` | v2 collection 重建入口 | 已实现 |
| 查询规划与检索 | `src/retrieval/query_pipeline.py` | 意图、多查询、RRF、重排、HyDE、父块回填 | 已实现；部分能力待验证 |
| Agent | `src/agent/graph_agent.py` | 检索、起草、审查、NLI、阻断和最终输出 | 已实现/安全路径已测试 |
| NLI | `src/safety/nli_grounding.py` | 对散文正文执行 entailment 验证 | 已实现；模型效果待扩充评估 |
| 审计存储 | `src/storage/*` | PostgreSQL repository 与 SQLite fallback | 已实现 |
| 数据迁移 | `migrations/*` | Alembic schema migration | 已实现 |
| 推理微服务 | `services/inference_api/*` | embedding、reranker、NLI HTTP API | 已实现；GPU 拓扑待验证 |
| 评估 | `src/evaluation/*` | 检索、规划链路、生成评估和回归门禁 | 已实现 |
| 应用 | `app.py` | Streamlit UI、依赖组装、结果展示和审计落库 | 已实现 |

依赖方向：

```text
app
  -> agent
     -> retrieval -> inference / Milvus
     -> safety    -> inference service or local model
  -> storage      -> PostgreSQL or SQLite fallback

indexing   -> inference / Milvus
evaluation -> retrieval / agent
all modules -> config
```

生产模块不得反向依赖 `src/evaluation`；检索层不得依赖 UI 或 Agent；provider-specific 细节不得进入业务节点。

## 5. 索引架构

### 5.1 语义父子切块

当前默认配置：

| 参数 | 默认值 |
|---|---:|
| Parent size | 1500 tokens |
| Child size | 400 tokens |
| Child overlap | 200 tokens |
| Chunk version | `semantic-v2` |

切块器在句子和段落边界上建立父块，再从父块生成有 overlap 的子块。父块负责保留完整上下文，子块负责提高检索粒度。

`chunk_id` 基于文档标识、稳定结构路径、块类型和内容生成；同一输入和配置应得到相同 ID。子块保存 `parent_id`，查询命中后可回填父块正文。

### 5.2 BGE-M3 双表示

`BGEM3EmbeddingClient` 对入库和查询使用同一模型接口，输出：

- dense vector：用于语义相似检索。
- lexical sparse weights：用于学习型词项匹配及语义扩展。

学习型 sparse token ID 与历史 blake2b sparse vector 不兼容，因此使用独立 v2 collection，禁止混写旧 collection。

### 5.3 Milvus collection

当前默认 collection：

```text
regtech_compliance_chunks_v2
```

关键字段包括：

- `chunk_id`
- `parent_id`
- `chunk_type`
- `source_file`
- `page_number`
- `text`
- `metadata_json`
- dense vector
- sparse vector

默认索引和度量：

| 通道 | 索引 | 度量 |
|---|---|---|
| Dense | HNSW | COSINE |
| Sparse | SPARSE_INVERTED_INDEX | IP |

改变 embedding revision、稀疏编码、向量维度、切块 ID 算法或 schema 时，必须创建新 collection 并重新建立评估基线。

## 6. 查询规划与检索链路

### 6.1 规划阶段

`ComplianceRetrievalPipeline.retrieve()` 执行：

1. 根据查询特征决定是否进行多查询扩展。
2. 意图分类与多查询生成通过 `asyncio.gather` 并行等待。
3. Planner 结果进入带 TTL 的进程内缓存。
4. 原查询始终保留并获得 `2.0` 的默认查询权重。

Planner 使用角色化配置。当前运行环境可通过 OpenAI-compatible 接口接入 DeepSeek；生产目标由本地 Qwen2.5-7B-Instruct/vLLM 承担规划任务。

### 6.2 双路检索与融合

每个查询只检索 `chunk_type == "child"`：

1. BGE-M3 批量生成 dense 与 sparse 表示。
2. dense/sparse 检索在受 semaphore 限制的并行任务中执行。
3. 各查询、各通道结果以 weighted RRF 融合。
4. 默认 dense:sparse 权重保持 `1:1`。
5. RRF 结果截取候选后进入可选 reranker。

直接检索的 `8:1` 权重实验没有在完整规划链路中超过当前基线，因此没有成为默认配置。

### 6.3 Reranker 与条件 HyDE

Reranker 可通过本地模型或独立服务执行。只有 reranker 存在且产生可比较分数时，HyDE 才允许根据 top score 和 margin 触发。

```text
无 reranker / 无有效 score -> HyDE 禁用
score 或 margin 低于阈值     -> 生成 HyDE 假设文档并重新检索
score 充分                    -> 跳过 HyDE
```

BGE-Reranker-Large 在当前本机 CPU 环境中单查询耗时超过 8 分钟，75 条全量端到端验证尚未完成。因此 reranker 与条件 HyDE 目前是“已实现、待 GPU 全量验证”，不得描述为已验证的默认收益。

### 6.4 父块回填

重排或 RRF 截断完成后，系统按 `parent_id` 批量读取父块。父块内容替代子块正文进入生成上下文，同时保留：

- 命中 child 的 `chunk_id`
- `parent_id`
- 原始 score
- `matched_child_id`
- 来源文件和页码

同一父块只回填一次，避免多个相邻子块占用生成上下文。

## 7. LLM 接入架构

### 7.1 通用 client

业务层依赖统一 `LLMClient` 协议，不直接调用 provider SDK。当前支持：

- OpenAI-compatible：DeepSeek、vLLM、TGI 或其他兼容服务。
- Anthropic adapter：为历史配置保留的兼容路径，不是目标本地部署架构。

每个角色独立配置 `provider`、`model`、`base_url`、API key、temperature、max tokens 和 timeout，因此切换 provider 不需要修改检索或 Agent 业务代码。

### 7.2 角色拆分

| 角色 | 当前可配置运行方式 | 生产目标 |
|---|---|---|
| Planner | DeepSeek/OpenAI-compatible | Qwen2.5-7B-Instruct on vLLM |
| Draft | DeepSeek/OpenAI-compatible | Qwen2.5-72B-Instruct-AWQ on vLLM |
| Auditor | DeepSeek/OpenAI-compatible | Qwen2.5-72B-Instruct-AWQ on vLLM |
| Judge | DeepSeek/OpenAI-compatible | Qwen2.5-72B-Instruct-AWQ on vLLM |

本机 8 GB 内存不能运行 70B/72B 4-bit 模型；72B AWQ 双 GPU 配置仅为生产目标。

## 8. LangGraph 审计闭环

当前状态机：

```text
retrieve
   |
   v
generate_draft <----------------+
   |                             |
   v                             |
auditor_review -- rejected ------+
   |
   +-- approved --> grounding_check --> finalize
   |
   +-- error -------------------------> error_handler
   |
   +-- max iterations ---------------> finalize(blocked)
```

### 8.1 审查规则

- Auditor 输出必须是 JSON object。
- `approved` 必须存在且为 boolean。
- 非法 JSON、字段缺失、调用异常或超时均进入 error 状态，不得默认放行。
- rejected 且仍有剩余轮次时，反馈返回 Draftee 修订。
- 最多三轮，达到上限后状态转为 `max_iterations`。

### 8.2 用户输出边界

只有 `audit_status == "approved"` 的草稿可以作为答案返回。

未通过最大轮次时，草稿被阻断，用户仅看到：

```text
本报告未经审计通过
```

检索为空时返回证据不足信息。错误路径返回 error 状态、空 claims 和空 citations，不把内部未审计草稿泄漏给用户。

### 8.3 两层据实性校验

1. Claim validation：每条结构化 claim 的 `source_ids` 必须映射到实际检索块；无来源或来源无效的 claim 被移除。
2. NLI grounding：对散文正文逐句执行多语言 entailment 检查；不通过时拒绝，NLI 调用错误时 fail-closed。

## 9. 审计存储

生产式容器通过 `DATABASE_URL` 使用 PostgreSQL 16，schema 由 Alembic 管理。repository 保存请求、响应状态、迭代次数、token、反馈与证据等审计信息。

`create_transaction_repository()` 的当前选择规则为：

```text
DATABASE_URL 存在 -> PostgresTransactionRepository
DATABASE_URL 缺失 -> SQLite TransactionRepository
```

SQLite 是本地兼容回退，不是目标生产审计存储。生产环境必须设置 `DATABASE_URL` 并先执行 `alembic upgrade head`。

## 10. 部署拓扑

Docker Compose 当前定义：

- `postgres`
- `etcd`
- `minio`
- `milvus`
- `attu`
- `streamlit`
- `planner-llm`
- `generation-llm`
- `embedding` / `embedding-cpu`
- `reranker` / `reranker-cpu`
- `nli`
- `migrate`

`app` 和 `gpu` profiles 描述生产式目标拓扑。GPU 模型服务、资源分配及全链路性能仍需在目标硬件验证。详细状态和命令见 [`PRODUCTION_UPGRADE.md`](PRODUCTION_UPGRADE.md)。

## 11. 评估与质量门禁

### 11.1 当前检索基线

基线文件：`src/evaluation/baseline_metrics.json`

| 指标 | 数值 |
|---|---:|
| Hit@1 | 0.5467 |
| Hit@3 | 0.7600 |
| Hit@5 | 0.8667 |
| Recall@10 | 0.8933 |
| MRR | 0.6734 |

元数据：

- collection：`regtech_compliance_chunks_v2`
- query count：75
- 查询集：25 个规范问题的 EN、ZH、HK-mixed 变体
- mode：conditional multi-query + weighted RRF + parent backfill
- reranked：false
- generated at：2026-07-12

以上结果不包含 BGE-Reranker-Large 全量重排，不得作为 reranker 或 HyDE 的效果证明。

### 11.2 CI

GitHub Actions 当前包含：

- Python 3.11 compile check。
- `pytest -q` 单元/集成测试。
- 当 `reports/candidate_metrics.json` 存在时运行 regression gate。

评估 gate 目前是条件执行，不等同于 required branch check。将其升级为强制门禁前，必须确保 candidate 由相同语料、collection 和配置生成。

## 12. 当前限制与风险

1. Reranker 的 CPU 延迟不满足交互式使用，全量验证依赖 GPU 服务。
2. 没有 reranker score 时 HyDE 必须禁用，因此当前基线不是完整 reranker + HyDE 链路。
3. DeepSeek 属于外部 API，真实客户数据和内部材料不得在未批准情况下发送。
4. 本地 Qwen 72B 数据驻留方案未在本机运行验证。
5. PostgreSQL 代码和 Compose 已具备，但本地未设置 `DATABASE_URL` 时仍会回退 SQLite。
6. NLI 模型需要更大规模中英文/HK-mixed 标注集校准阈值。
7. 当前 75 条检索集来自 25 个规范问题，后续应拆分调参与独立验收集。

## 13. 当前目录结构

```text
RegTech-RAG-Assistant/
├── PROJECT_RULES.md
├── app.py
├── config/
│   └── settings.py
├── src/
│   ├── agent/
│   ├── evaluation/
│   ├── indexing/
│   ├── inference/
│   ├── retrieval/
│   ├── safety/
│   └── storage/
├── services/
│   ├── inference_api/
│   └── migration/
├── migrations/
├── tests/
├── data/raw_pdfs/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PRODUCTION_UPGRADE.md
│   └── superpowers/specs/ARCHITECTURE_LEGACY.md
├── docker-compose.yml
├── alembic.ini
└── .github/workflows/ci.yml
```

早期编号脚本保留在 `archive/`，不得作为当前应用入口或架构依据。

## 14. 架构演进顺序

在 `PROJECT_RULES.md` 规定的架构评审和单模块流程下，推荐顺序为：

1. 持续回归 fail-closed、最大轮次阻断与错误输出边界。
2. 为 DeepSeek 与 vLLM 角色路由补齐统一契约测试。
3. 固化 BGE-M3 revision、切块配置和 collection build manifest。
4. 在 GPU 环境完成 BGE-Reranker-Large 全量评估。
5. 基于有效 reranker score 完成条件 HyDE 消融和延迟评估。
6. 验证 PostgreSQL 备份恢复、GPU 推理服务和 required CI gate。
7. 扩充独立合规验收集并加入人工审阅。

涉及 provider 协议、collection schema、安全状态机、评估阈值或生产拓扑的重大变化必须通过 ADR。
