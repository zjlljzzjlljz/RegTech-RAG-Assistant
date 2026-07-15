# RegTech RAG Assistant 项目开发规则

> 文档性质：本项目唯一权威开发规范（Single Source of Truth, SSOT）
> 生效日期：2026-07-15
> 当前版本：1.0
> 适用范围：需求分析、架构设计、编码、测试、评估、文档、部署、迁移、Git 与发布活动

## 0. 权威性与使用方式

### 0.1 单一权威来源

1. 本文件是项目开发流程和工程约束的唯一规范性来源。
2. `docs/ARCHITECTURE.md`、`docs/PRODUCTION_UPGRADE.md`、ADR、README、`DEVELOPMENT_LOG.md` 和代码注释用于描述架构、决策、状态与实现细节，不得覆盖本文件。
3. 当资料互相冲突时，按以下优先级处理：
   1. 法律、监管、数据安全和 fail-closed 安全约束。
   2. 本文件及其已生效的追加修订。
   3. 已接受的 ADR。
   4. 当前架构文档与生产升级文档。
   5. `DEVELOPMENT_LOG.md`。
   6. README、代码注释和其他说明材料。
4. `docs/ARCHITECTURE.md` 中的 ChromaDB、MiniLM、Claude、早期编号脚本和旧切块参数属于历史方案，除非后续 ADR 明确恢复，否则不是当前实现依据。
5. `docs/PRODUCTION_UPGRADE.md` 同时包含已实现能力和目标拓扑。任何标记为“规划中”“待验证”的内容不得在代码、文档或简历中表述为已生产验证。

### 0.2 规则变更方式

1. 本文件长期维护，历史原则不得删除、静默改写或覆盖。
2. 后续规则变更只能追加到“规则修订记录”，必须包含日期、提出原因、影响范围、替代关系和批准状态。
3. 新规则需要替代旧规则时，旧规则必须保留，并写明“由修订 R-XXX 替代”；不得直接删除旧文本。
4. 未经用户明确确认的修订状态为“提议”，不具有约束力。
5. 发生紧急安全修复时可先执行最小化风险控制，但必须在同一开发周期补充 ADR、测试和修订记录。

### 0.3 当前仓库治理缺口

截至 2026-07-15，仓库中尚未发现以下治理材料：

- 根目录 `DEVELOPMENT_LOG.md`
- 根目录 `ADR/` 目录及 ADR 文件
- 根目录 README

在未来开发开始前必须报告这些缺失项。除非用户明确要求，不得借本规则文件创建任务擅自补建其他文件。

---

# 1. 项目目标

## 1.1 项目定位

本项目是面向香港银行 AML/CFT 合规场景的 RegTech RAG 助手原型。系统以 HKMA 公开监管材料为证据源，为中英文及香港混合表达查询提供可追溯的检索增强回答，并通过起草、审计、据实性校验和审计日志降低无依据陈述进入最终结果的风险。

该系统是辅助研究与合规分析工具，不是法律意见、监管裁决或自动审批系统。任何对外结果必须保留来源、验证状态和必要的人工复核提示。

## 1.2 核心功能

1. 解析和索引 HKMA AML/CFT PDF 文档。
2. 使用语义父子切块、稳定 ID、重叠窗口和父块回填保留上下文。
3. 使用 BGE-M3 生成稠密向量与原生学习型稀疏权重。
4. 在 Milvus 中进行稠密、稀疏双路检索并以 RRF 融合。
5. 进行意图识别、多查询扩展和有条件的 HyDE 查询规划。
6. 使用 LangGraph 执行起草、审计、修订闭环，最多三轮。
7. 对结构化 claims 和散文正文进行来源核验与 NLI 据实性校验。
8. 对审查解析错误、模型错误、NLI 错误和未通过草稿执行 fail-closed。
9. 将审计记录持久化到 PostgreSQL，并通过 Alembic 管理 schema。
10. 使用检索评估、生成评估和 CI 回归门禁量化系统质量。

## 1.3 当前技术栈

| 层级 | 当前实现或规范 |
|---|---|
| 应用入口 | Python 应用，当前入口为 `app.py` |
| Agent 编排 | LangGraph 起草与审计状态机 |
| LLM 接入 | provider-neutral、OpenAI-compatible 通用 client；当前可配置 DeepSeek API |
| 目标本地模型 | 规划模型 Qwen2.5-7B-Instruct；生成/审计模型 Qwen2.5-72B-Instruct-AWQ 或等价 70B 级模型 |
| 目标推理引擎 | vLLM；TGI 仅可通过 ADR 批准后替换 |
| Embedding | BGE-M3 dense + native lexical sparse |
| 向量数据库 | Milvus，当前 v2 collection 为 `regtech_compliance_chunks_v2` |
| 融合 | RRF；当前生产候选权重 dense:sparse = `1:1`，原查询权重 `2.0` |
| 重排 | BGE-Reranker-Large 接口已纳入架构，但本机 CPU 全量端到端验证未完成 |
| 据实性校验 | 多语言 NLI 服务及 claims/source 交叉核验 |
| 审计存储 | PostgreSQL + Alembic；SQLite WAL 属于历史原型实现 |
| 部署 | Docker Compose；应用、Milvus、PostgreSQL及可选 GPU 推理服务 |
| 测试 | pytest |
| 评估 | Hit@K、Recall@K、MRR、RAGAS 及回归门禁 |

## 1.4 当前已验证基线

以下数值是当前可引用的检索基线，不得与其他 collection、模型 revision 或参数的结果混用：

| 指标 | 当前结果 |
|---|---:|
| Hit@1 | 0.5467 |
| Hit@3 | 0.7600 |
| Hit@5 | 0.8667 |
| Recall@10 | 0.8933 |
| MRR | 0.6734 |

评估集包含 25 个规范问题的 EN、ZH、HK-mixed 变体，共 75 条检索记录。最近一次已知自动化测试结果为 `41 passed`。这些结果只在评估配置、语料、collection、模型版本和代码未发生实质变化时有效。

加权 RRF 实验中，直接检索的 dense:sparse `8:1` 得到 Recall@10 `0.88`、MRR `0.6247`；端到端规划链路在 `8:1` 下得到 Recall@10 `0.88`、MRR `0.6106`，均未超过当前基线，因此默认融合权重保持 `1:1`。

## 1.5 明确未完成的验证

1. BGE-Reranker-Large 在当前本机 CPU 环境中单查询超过 8 分钟，尚未完成 75 条全量端到端验证。
2. 当没有有效 reranker score 时，条件 HyDE 必须保持禁用；当前不得宣称 HyDE 已在完整生产链路验证有效。
3. 本机 8 GB 内存不能承载 70B/72B 4-bit 推理；72B AWQ 双 GPU 拓扑是目标生产配置，不是当前本机运行状态。
4. DeepSeek API 是当前低成本外部 provider 配置，不等同于 HKMA 数据驻留目标已经达成。
5. `baseline_metrics.json` 的任何占位 schema 只有在 v2 collection 上重新生成有效指标后才能作为 CI 必选门禁。

---

# 2. 开发原则

## 2.1 正确性优先

1. 合规回答的证据正确性、安全路由和审计可追溯性高于响应速度、代码简短和界面效果。
2. 无法验证的内容必须拒绝、降级或明确标记，不得猜测补全。
3. 任何可能把未通过审计内容返回用户的改动均视为高风险改动。

## 2.2 可维护性优先

1. 优先使用现有模块边界、类型、配置系统和测试模式。
2. 业务代码不得直接依赖具体 LLM provider SDK。
3. 结构化数据使用结构化解析与校验，不使用脆弱的字符串拼接代替 schema。
4. 配置、代码和评估产物必须能说明“使用了哪个模型、版本、collection 和参数”。

## 2.3 小步迭代

1. 每次变更只解决一个明确问题，并控制可回滚范围。
2. 先建立失败测试或可复现基线，再修改实现。
3. 大规模迁移必须拆成兼容层、数据迁移、流量切换和旧路径清理等独立步骤。

## 2.4 单模块开发

1. 一次只能开发一个业务模块。
2. 当前模块达到完成标准后，才能进入下一模块。
3. 禁止同时开发多个互不依赖的模块。
4. 当前模块所需的测试、文档和迁移属于该模块的完成工作，不视为并行开发其他模块。

## 2.5 禁止事项

1. 禁止 Vibe Coding，即在未理解现有代码、数据流、约束和测试的情况下凭感觉修改。
2. 禁止一次生成或重写整个项目。
3. 禁止跳步开发、先实现后补设计，或先合并后补测试。
4. 禁止为了通过测试而降低 fail-closed、删除断言或放宽评估阈值。
5. 禁止在没有数据支持时宣称性能、准确率、幻觉消除或生产可用性。
6. 禁止顺手重构无关模块或覆盖用户尚未提交的改动。

---

# 3. 开发前强制协议

## 3.1 必读顺序

每次新对话、上下文丢失、长期中断或开始新模块前，必须依次阅读：

1. 用户提供的项目大纲和当前任务。
2. 根目录 `PROJECT_RULES.md`。
3. `docs/ARCHITECTURE.md`。
4. 根目录 `DEVELOPMENT_LOG.md`。
5. 根目录 `ADR/*` 中所有已接受或相关的 ADR。
6. 与当前模块直接相关的代码、测试、配置和最近 Git diff。

文件不存在时必须明确报告，不得假装已阅读，也不得未经请求自动创建。

## 3.2 架构评审门禁

完成必读材料后不得立即写代码。必须先输出以下架构评审：

```text
项目理解：

核心模块：

模块依赖关系：

潜在风险：

建议优化项：

推荐开发顺序：

是否发现架构问题：
```

架构评审必须明确：当前事实、假设、未验证项、预期改动范围、测试计划和数据迁移影响。输出后等待用户确认；在确认前不得修改应用代码、配置、数据库 schema 或部署文件。

纯文档治理任务可只修改用户明确指定的治理文档，但不得借此启动业务开发。

---

# 4. 开发流程

所有模块严格遵循：

`分析 → 设计 → 实现 → 测试 → 文档更新 → Git 提交阶段 → 等待确认`

不得跳过、倒置或合并阶段。

## 4.1 分析

1. 定义用户问题、验收标准、影响模块和非目标。
2. 阅读现有实现、测试、配置、数据 schema 和相关 ADR。
3. 检查工作区状态，识别用户已有改动并保留。
4. 对 RAG 改动记录现有 collection、模型 revision、评估集和基线。
5. 对高风险安全路径列出失败模式和 fail-closed 行为。

## 4.2 设计

1. 给出接口、数据流、状态变化、错误处理和兼容策略。
2. 明确是否需要 ADR、数据库迁移、collection 版本升级或环境变量变化。
3. 定义单元、集成、回归和性能测试。
4. 明确回滚方法和观测指标。
5. 设计获得用户确认后才进入实现。

## 4.3 实现

1. 只修改设计范围内的单一模块及其必要测试、文档。
2. 保持 provider、模型和基础设施通过配置注入。
3. 不删除旧路径，除非迁移完成、有回滚方案且用户批准。
4. 任何安全失败必须返回显式状态，不得吞掉异常后默认成功。

## 4.4 测试

1. 先运行变更模块的最小测试，再运行相关集成测试，最后运行完整测试集。
2. 检索或生成逻辑变化必须运行对应评估，不得只依赖单元测试。
3. 部署变化必须验证容器健康、依赖可达性、持久化和重启行为。
4. 无法运行的测试必须说明原因、影响和后续验证命令。

## 4.5 文档更新

1. 更新根目录 README 中的运行方式、配置或用户可见行为。
2. 追加更新 `DEVELOPMENT_LOG.md`，记录日期、改动、测试、指标和未完成项。
3. 架构或数据流变化必须更新 `docs/ARCHITECTURE.md`。
4. 生产拓扑变化必须更新 `docs/PRODUCTION_UPGRADE.md`。
5. 重大决策必须新增 ADR，不得只写在聊天或 commit message 中。

## 4.6 Git 提交阶段

1. 检查 diff，只包含当前模块改动。
2. 生成符合本文件规范的 commit message。
3. 未经用户明确授权不得擅自提交、推送、rebase 或改写历史。
4. 用户已授权提交时，提交前必须确认测试与文档完成。

## 4.7 等待确认

报告变更、验证结果、已知限制、文件列表和建议 commit message，然后等待用户确认是否进入下一模块。

---

# 5. 模块开发规则

## 5.1 模块边界

当前核心模块及责任如下：

| 模块 | 主要路径 | 责任 |
|---|---|---|
| 配置 | `config/settings.py` | 环境变量、模型、collection、阈值和服务地址 |
| LLM 抽象 | `src/inference/llm_client.py` | OpenAI-compatible provider-neutral 调用、角色模型路由 |
| 语义切块 | `src/indexing/semantic_chunker.py` | 句段边界、overlap、父子关系、稳定 ID |
| 稀疏编码 | `src/indexing/sparse_tokenizer.py` | BGE-M3 原生稀疏向量的一致编码接口 |
| 索引 | `src/indexing/milvus_ingest.py`、`rebuild_milvus.py` | schema、写入、collection 版本和重建 |
| 查询规划/检索 | `src/retrieval/query_pipeline.py` | 意图、多查询、条件 HyDE、双路检索、RRF、父块回填、重排 |
| Agent | `src/agent/graph_agent.py` | 起草、审计、修订、最大轮次和返回状态 |
| 安全校验 | `src/safety/nli_grounding.py` | claims 与散文正文的 entailment/grounding |
| 审计存储 | `src/storage/*`、`migrations/*` | 事务日志、PostgreSQL repository、Alembic |
| 推理服务 | `services/inference_api/*` | embedding、sparse、reranker、NLI 服务接口 |
| 评估 | `src/evaluation/*` | 检索、规划链路、生成质量和回归门禁 |
| 部署 | `docker-compose.yml`、服务 Dockerfile | 服务拓扑、健康检查、资源与持久化 |

## 5.2 模块依赖方向

允许的主要依赖方向：

```text
app
  -> agent
     -> retrieval -> inference client / inference services -> Milvus
     -> safety    -> inference services
     -> storage   -> PostgreSQL

indexing -> inference services -> Milvus
evaluation -> retrieval / agent / metrics
all modules -> config
```

约束：

1. `config` 不得反向依赖业务模块。
2. `retrieval` 不得依赖 `agent` 或 UI。
3. `safety` 不得依赖最终展示层。
4. provider SDK 只能封装在 inference 边界内。
5. evaluation 可调用生产模块，但生产模块不得导入 evaluation 代码。
6. storage repository 不得包含提示词、检索或 UI 逻辑。

## 5.3 模块完成标准

一个模块只有同时满足以下条件才算完成：

1. 功能按已确认设计完成。
2. 新增及相关既有测试全部通过。
3. 根目录 README 已更新；若 README 缺失，先报告并取得补建批准。
4. `DEVELOPMENT_LOG.md` 已追加更新。
5. `docs/ARCHITECTURE.md` 已按实际变化更新。
6. 已生成符合规范的 commit message。
7. 需要时已完成 ADR、迁移说明、评估报告和回滚说明。
8. 用户已收到结果并确认是否继续。

---

# 6. 架构规范

## 6.1 配置与依赖注入

1. 模型名、`base_url`、API key、超时、重试、collection、阈值、融合权重和服务地址必须配置化。
2. 密钥只从环境变量或 secret manager 读取，不得提交到仓库、日志、fixture 或文档。
3. 业务代码使用角色语义选择模型，例如 planner、drafter、auditor、judge，不直接硬编码 provider 模型名。
4. 配置必须有类型、默认值说明和启动时校验；安全关键配置缺失时启动失败。

## 6.2 LLM provider 抽象

1. 所有模型调用通过 `src/inference/llm_client.py` 或其后继统一接口。
2. client 必须保持 OpenAI-compatible `base_url` 可配置，使 DeepSeek、vLLM 或其他兼容服务切换时不修改业务代码。
3. 查询规划默认使用 7B 级小模型；起草、审计和评估 judge 使用能力更强的大模型。
4. provider-specific 参数必须在 adapter 层映射，不得泄漏到检索或 agent 状态。
5. 必须设置连接超时、读取超时、有限重试和可观测错误；禁止无限重试。
6. Prompt cache key 必须包含模型、提示词版本、输入哈希和影响输出的参数，不得跨模型误复用。

## 6.3 Agent 状态机

1. LangGraph state 使用显式类型，状态转换可测试、可审计。
2. 起草与审查最多三轮；轮次不得由模型自行修改。
3. 审查结构化输出必须通过 schema 校验。
4. 解析错误、字段缺失、模型异常、超时或不确定状态统一 fail-closed。
5. 达到最大轮次且仍未通过时，外部结果必须阻断并显示 `本报告未经审计通过`。
6. 未通过草稿可保留在内部审计记录中，但不得作为已批准答案返回。

## 6.4 数据与服务边界

1. Milvus 保存检索向量、块文本和必要元数据，不承担审计事务存储。
2. PostgreSQL 保存追加式审计记录；schema 变更必须使用 Alembic。
3. 推理服务只暴露版本化 API 和健康检查，不持有业务会话状态。
4. 容器必须设置资源限制、健康检查、持久卷和明确的启动依赖。
5. 生产默认不得依赖隐式本地 fallback；`PREFER_LOCAL_INFERENCE_FALLBACK=false` 时失败必须可见。

---

# 7. 代码规范

## 7.1 Python 规范

1. 遵循项目现有 Python 版本、格式化和 lint 约定；新增公共接口必须有类型注解。
2. 函数保持单一职责；复杂分支拆为可独立测试的纯函数。
3. 结构化 LLM 输出、配置、数据库记录和 API payload 使用明确 schema。
4. 捕获具体异常；禁止裸 `except` 和吞掉异常后返回成功。
5. 日志使用结构化字段，至少包含 request/trace ID、阶段、模型或服务、耗时和状态；不得记录密钥或完整敏感提示词。
6. 注释解释原因、约束和非显然行为，不复述代码。
7. 默认使用 ASCII；监管原文、水印和中英文测试数据可使用 UTF-8。

## 7.2 接口规范

1. 公共函数和服务 API 的输入、输出、异常与幂等性必须稳定并有测试。
2. 新增字段优先向后兼容；删除或改变语义需要迁移计划与 ADR。
3. 所有推理响应必须携带模型标识或可追溯版本信息。
4. 安全结果不得用模糊布尔值表达，至少区分 approved、rejected、error、blocked。

## 7.3 错误处理

1. 安全与合规路径错误默认拒绝。
2. 检索无结果不得伪造上下文；返回可解释的证据不足状态。
3. 外部服务错误必须保留根因链并转换为稳定的领域错误。
4. 重试只用于明确的瞬时错误，并使用有限次数和退避。

---

# 8. 测试规范

## 8.1 测试层级

1. 单元测试：切块边界、ID、融合、门控、解析、状态转换和错误分支。
2. 集成测试：Milvus schema/检索、PostgreSQL repository、推理 API 契约和 LangGraph 路径。
3. 端到端测试：从查询规划到检索、生成、审计、NLI 和审计落库。
4. 回归评估：75 条检索集及维护中的生成评估集。
5. 性能测试：p50/p95 延迟、模型吞吐、GPU/CPU 内存和超时行为。

## 8.2 强制安全测试

至少覆盖：

- auditor 返回非法 JSON 时拒绝。
- approval 字段缺失时拒绝。
- auditor、NLI 或 provider 超时时拒绝。
- 最大三轮仍未通过时阻断并显示水印。
- 未通过草稿不进入成功响应。
- claims 与散文正文出现无证据陈述时被标记或拒绝。
- 审计记录在失败路径仍保留必要状态和证据元数据。

## 8.3 检索测试

1. 入库和查询必须使用同一 BGE-M3 模型、revision、稀疏编码规则和归一化方式。
2. 稳定 `chunk_id` 在相同输入和配置下必须可复现。
3. `parent_id` 必须能解析到有效父块，父块回填不得改变命中归因。
4. dense、sparse、RRF、规划链路和 rerank 必须可独立消融。
5. 权重或阈值只能依据完整评估集结果调整，不得根据少量样例手调。

## 8.4 测试结果规则

1. 不得只报告“测试通过”，必须记录命令、通过数量和跳过/失败项。
2. 网络、GPU、API 费用或硬件限制导致未运行时，必须明确标记“未验证”。
3. flaky 测试不得简单重跑掩盖；必须定位原因或隔离并登记风险。
4. 不得使用真实 API key 进入测试产物。

---

# 9. RAG 与 RegTech 专项规范

## 9.1 语料与切块

1. 原始监管 PDF 保持只读和来源可追溯，记录文件名、来源、发布日期、版本和内容哈希。
2. 子块目标约 400 token，使用句子/段落边界，配置 overlap；当前目标 overlap 为 200 token。
3. 父块保留更完整章节上下文；子块用于命中，父块用于回填和生成上下文。
4. 标题、章节、页码和文档元数据必须尽可能保留。
5. `chunk_id` 由规范化文档标识、结构路径、内容或稳定边界生成；不得使用随机 ID。
6. 改变切块算法、tokenizer 或 ID 算法必须新建 index/collection 版本并执行迁移评估。

## 9.2 稀疏与稠密向量一致性

1. BGE-M3 dense 和 native lexical sparse 必须来自固定模型 revision。
2. 入库侧和查询侧必须共享同一编码实现、版本和预处理。
3. 学习型 sparse token ID 与历史 blake2b sparse 不兼容，禁止写入同一 collection。
4. collection schema 或 embedding 维度变化必须创建新 collection，禁止原地破坏性迁移。
5. 每次重建记录语料哈希、模型 revision、切块配置、schema 版本和构建时间。

## 9.3 检索与融合

1. 默认 collection 为 `regtech_compliance_chunks_v2`。
2. 默认 RRF dense:sparse 权重为 `1:1`；原查询在多查询融合中的权重为 `2.0`。
3. 任何权重变更必须同时比较 Hit@K、Recall@10、MRR、延迟和失败样例。
4. 当前 `8:1` 实验结果低于基线，不得作为默认配置。
5. 父块回填发生在子块命中后，必须去重并保留原命中分数、rank 和 provenance。
6. 重排不得丢失来源元数据；无合法 score 时必须显式标记 unavailable。

## 9.4 查询规划与 HyDE

1. 意图分类、多查询生成和可独立执行的规划调用可并行，但必须保证结果顺序和错误语义确定。
2. 多查询扩展必须支持英文、中文和香港混合表达，并保留原查询。
3. HyDE 是低置信度补救路径，不得无条件执行。
4. HyDE 门控只使用可比较、校准过的 reranker/retrieval score；没有有效 score 时保持禁用。
5. HyDE 开启前必须完成有/无 HyDE 消融，证明质量收益能够覆盖延迟和 API 成本。
6. 规划调用的 cache 不得跨租户泄漏数据，敏感查询不得写入不受控缓存。

## 9.5 重排

1. BGE-Reranker-Large 必须作为独立可观测阶段，记录输入候选数、输出分数、耗时和模型 revision。
2. CPU 单查询超过 8 分钟的当前现状不满足交互式 SLO；全量验证应使用 GPU 推理服务或明确的离线批处理。
3. 重排启用必须有超时、批处理上限和失败降级策略。
4. 降级到未重排结果时必须保留状态，且不得触发依赖 reranker score 的 HyDE 门控。

## 9.6 生成、引用与据实性

1. 最终事实性陈述必须能映射到检索证据或被明确标记为无法确认。
2. 引用必须保留文档、页码/章节、chunk 和 parent provenance。
3. post-hoc claims 校验不能替代散文正文 NLI；两者均需覆盖。
4. NLI 阈值、模型和语言覆盖必须版本化并通过中英文样例校准。
5. 禁止使用“消除幻觉”等绝对表述；只能报告特定评估集上的量化结果。

## 9.7 合规、安全与数据驻留

1. 默认最小化向外部 LLM 发送的数据，只发送完成任务所需的查询和证据片段。
2. 真实客户信息、交易数据、身份数据和内部监管材料不得发送到未经批准的外部 API。
3. 本地 vLLM 是数据驻留目标方案，但只有在部署、访问控制、日志和备份完成验证后才能宣称合规落地。
4. API key 泄露后立即撤销并轮换；新 key 只写入 `.env` 或 secret manager。
5. 日志和评估产物必须脱敏，设置保留期限和访问权限。
6. 所有拒绝、阻断、模型失败和人工覆盖操作必须可审计。

---

# 10. 评估与回归门禁

## 10.1 评估集管理

1. 规范问题、语言变体和 ground truth 分开版本化。
2. 当前检索套件为 25 个规范问题派生的 75 条 EN/ZH/HK-mixed 查询。
3. 扩充评估集时必须保留旧集合，记录新增来源、标注方法和评审人。
4. 调参集与最终验收集应分离，避免对 75 条集合过拟合。

## 10.2 检索指标

至少报告：Hit@1、Hit@3、Hit@5、Recall@10、MRR、每阶段延迟和失败率。涉及父块回填时还需区分子块命中和父文档命中。

## 10.3 生成指标

至少报告：Faithfulness、Answer Relevancy、Context Recall、引用覆盖率、审计通过率、最大轮次阻断率、NLI contradiction/unknown 比例和人工抽检结果。

LLM-as-judge 必须记录 judge 模型、提示词版本和采样参数；同一对比实验使用相同 judge 配置。

## 10.4 CI 门禁

1. 候选结果必须与同一 collection、语料和评估集上的有效 baseline 比较。
2. placeholder baseline 不得启用为 required check。
3. 指标阈值变更需要 ADR 和用户批准，禁止为使 CI 通过而临时下调。
4. 安全测试失败、Recall/MRR 超过允许退化、schema 不兼容或迁移失败时阻断合并。
5. CI 中需要外部付费 API 的评估应使用受控 nightly/manual job；PR 门禁使用确定性测试或批准的本地服务。

---

# 11. 文档规范

## 11.1 文档职责

| 文档 | 职责 |
|---|---|
| `PROJECT_RULES.md` | 唯一开发规范与治理原则 |
| README | 使用者入口、安装、配置、启动和常用命令 |
| `docs/ARCHITECTURE.md` | 当前架构、模块、数据流和已验证设计 |
| `docs/PRODUCTION_UPGRADE.md` | 生产目标、迁移步骤、硬件与部署差距 |
| `DEVELOPMENT_LOG.md` | 按日期追加的实际开发、测试、指标和未完成事项 |
| `ADR/*` | 重大且长期有效的架构决策及替代关系 |

## 11.2 写作要求

1. 区分“当前实现”“已验证”“实验结果”“规划中”和“历史方案”。
2. 命令、路径、模型名、collection 和指标必须精确且可复现。
3. 不使用无法验证的营销表述，不把测试环境结果描述成生产 SLO。
4. 代码变更与文档在同一模块周期完成。
5. 旧架构内容若保留，必须明确标注历史状态和替代方案。

## 11.3 DEVELOPMENT_LOG 格式

未来创建后，每次只追加：

```markdown
## YYYY-MM-DD - 模块/任务

- 目标：
- 设计/ADR：
- 改动文件：
- 测试命令与结果：
- 评估指标：
- 已知限制：
- 下一步：
- Commit：
```

---

# 12. Git 规范

## 12.1 分支与变更范围

1. 一个分支或提交只承载一个模块或一个明确修复。
2. 不得提交 API key、`.env`、模型权重、数据库数据、缓存或大体积生成物。
3. 提交前检查 `git status` 和 diff，保留与当前任务无关的用户改动。
4. 禁止未经批准执行 `reset --hard`、强推、改写共享历史或删除用户分支。

## 12.2 Commit Message

采用 Conventional Commits 风格：

```text
<type>(<scope>): <summary>
```

允许类型：`feat`、`fix`、`refactor`、`test`、`docs`、`chore`、`perf`、`build`、`ci`、`revert`。

示例：

```text
fix(agent): block unaudited drafts after max iterations
feat(retrieval): add stable parent chunk backfill
docs(governance): establish project development rules
```

提交正文应说明原因、关键行为、测试结果和迁移影响。生成 commit message 不等于获得执行提交或推送的授权。

## 12.3 合并条件

1. 模块完成标准全部满足。
2. CI 和规定评估门禁通过。
3. ADR 与迁移已审阅。
4. 无密钥、敏感数据或无关文件。
5. 用户确认可以进入合并或下一模块。

---

# 13. ADR 规范

## 13.1 必须创建 ADR 的情况

- 更换向量数据库、LLM provider 协议或推理引擎。
- 修改 collection schema、embedding/sparse 模型或稳定 ID 算法。
- 修改 fail-closed、安全水印、审计状态或最大轮次行为。
- 修改默认融合权重、HyDE 门控依据或评估门禁阈值。
- 引入跨模块抽象、数据库迁移策略或生产拓扑变化。
- 弃用现有公共接口或历史数据。

## 13.2 文件与编号

ADR 放在根目录 `ADR/`，文件名格式：

```text
ADR/NNNN-short-title.md
```

编号递增且不复用。状态为 Proposed、Accepted、Superseded、Rejected 或 Deprecated。

## 13.3 ADR 模板

```markdown
# ADR-NNNN: 标题

- 日期：YYYY-MM-DD
- 状态：Proposed
- 决策者：
- 关联任务：

## 背景
## 决策驱动因素
## 备选方案
## 决策
## 影响
## 安全与合规影响
## 数据/迁移影响
## 测试与评估计划
## 回滚方案
## 替代关系
```

ADR 一经接受不得改写决策历史；新决策通过新 ADR supersede 旧 ADR。

---

# 14. 重构规范

1. 重构必须有明确目标：降低复杂度、消除真实重复、改善测试性或修复架构边界。
2. 行为保持型重构先建立 characterization tests，重构前后必须通过相同测试与评估。
3. 不得在功能开发中夹带大规模重构；必要时拆成独立模块周期。
4. 跨模块重构必须有 ADR、分阶段迁移和回滚方案。
5. provider、storage、retrieval 等核心接口变更应先提供兼容 adapter，再迁移调用方。
6. 删除 legacy 代码前必须确认无运行入口、无数据依赖、文档已更新且用户批准。
7. 重构不得改变安全默认值、评估阈值或数据保留行为，除非这些变化在设计中明确批准。

---

# 15. 部署、迁移与运维规范

## 15.1 Docker Compose

1. 当前生产式启动目标为：

   ```bash
   docker compose --profile app --profile gpu up -d
   ```

2. 72B AWQ 服务按双 GPU 设计；部署前必须按实际硬件调整 tensor parallelism 和资源预留。
3. 本机不满足 GPU/内存要求时，不得通过 swap 或无界 CPU 推理假装满足交互式服务要求。
4. 每个服务必须有健康检查、日志、超时和资源限制。

## 15.2 Collection 迁移

1. 学习型 sparse 与 blake2b legacy vector 不兼容，迁移必须使用新 collection。
2. 重建命令必须在固定语料、模型 revision 和配置下执行，并保存 manifest。
3. 新 collection 完成完整评估前不得切换默认流量。
4. 切换后保留旧 collection 至回滚窗口结束，再经批准清理。

## 15.3 PostgreSQL 迁移

1. 所有 schema 变化使用 Alembic revision，禁止手工直接改生产表。
2. 迁移必须验证 upgrade、downgrade 或明确的前向修复策略。
3. 审计记录采用追加式设计；修正通过新记录表达，不原地删除历史。
4. 备份、恢复、保留期限和访问控制必须在生产启用前验证。

## 15.4 推理服务

1. embedding、reranker、NLI 和 LLM 服务使用固定镜像与模型 revision。
2. API 契约版本化，启动时暴露模型和 revision 信息。
3. 记录 p50/p95 延迟、吞吐、错误率、队列长度和 GPU/CPU 内存。
4. 降级路径必须显式、可审计，且不能绕过安全审查。

---

# 16. 安全审查清单

每个模块完成前检查：

- 是否存在默认放行路径？
- 是否会在解析失败时返回未审计内容？
- 是否会记录或暴露 API key、个人数据或完整敏感上下文？
- 是否保持证据、引用、模型和提示词版本可追溯？
- 是否可能把测试/实验状态误报为生产成功？
- 是否包含超时、有限重试、资源上限和失败状态？
- 是否需要数据迁移、回滚或 retention 处理？
- 是否新增外部数据传输或改变数据驻留边界？
- 是否完成中英文及 HK-mixed 输入验证？
- 是否更新审计日志和安全测试？

任一关键问题无法回答时，不得进入 Git 提交阶段。

---

# 17. 推荐开发顺序

除非新的架构评审和 ADR 批准调整，项目演进顺序为：

1. 维持并回归验证 fail-closed、最大轮次阻断与水印。
2. 维持通用 LLM client，完成 DeepSeek 与 vLLM 角色路由的契约测试。
3. 固化 BGE-M3 sparse、语义切块和 v2 collection 的可复现 manifest。
4. 验证父块回填、GPU reranker 和条件 HyDE 的完整消融与延迟。
5. 完善 PostgreSQL 审计迁移、独立推理服务和 CI 评估门禁。
6. 扩充独立验收集，校准 NLI 与人工合规评审流程。

每一步只处理一个模块，并重新执行开发前架构评审和用户确认。

---

# 18. 新对话上下文恢复摘要

新对话读取本文件后，应恢复以下事实：

1. 当前系统不是早期 ChromaDB/MiniLM/Claude 原型；现行检索为 Milvus v2 + BGE-M3 dense/sparse + RRF。
2. LLM 通过 OpenAI-compatible 抽象接入，当前可用 DeepSeek；vLLM 7B/72B 是目标本地拓扑。
3. 起草/审计闭环必须 fail-closed，最多三轮，未通过结果显示 `本报告未经审计通过`。
4. 语义切块使用稳定 ID、overlap、parent-child 和查询时父块回填。
5. BGE-Reranker-Large 和条件 HyDE 尚未在本机完成全量端到端验证，不得作为已验证亮点。
6. 当前可引用检索基线为 Hit@1 0.5467、Hit@3 0.7600、Hit@5 0.8667、Recall@10 0.8933、MRR 0.6734；评估集 75 条，最近已知测试为 41 passed。
7. dense:sparse `8:1` 实验没有超过默认 `1:1`，因此不得擅自改权重。
8. PostgreSQL、Alembic、推理服务和 CI gate 已进入代码/目标架构，但生产就绪度必须按实际验证分别表述。
9. 开始任何代码工作前，必须完成第 3 章的架构评审并等待用户确认。
10. 一次只开发一个模块，完成测试、文档和 Git 阶段后再进入下一模块。

---

# 19. 规则修订记录（仅追加）

## R-001：建立项目单一权威开发规范

- 日期：2026-07-15
- 状态：已生效
- 原因：合并架构说明、生产升级要求和现行实现状态，为后续新对话提供可恢复、可审计的统一开发上下文。
- 影响范围：全项目。
- 替代关系：本规则不删除历史文档；当历史文档与本规则冲突时，以第 0.1 节定义的优先级为准。
- 批准依据：用户要求创建 `PROJECT_RULES.md` 并将其作为 Single Source of Truth。

后续修订必须从 R-002 开始追加，不得修改或删除 R-001。

## R-002：分离当前架构与历史原型文档

- 日期：2026-07-15
- 状态：已生效
- 原因：原 `docs/ARCHITECTURE.md` 混合了 ChromaDB、MiniLM、Claude 和早期编号脚本等历史原型内容，容易与当前 Milvus、BGE-M3、OpenAI-compatible client、LangGraph 和 PostgreSQL 架构混淆。
- 影响范围：架构文档定位与开发前上下文读取。
- 替代关系：补充并澄清第 0.1 节第 4 项。历史内容现位于 `docs/superpowers/specs/ARCHITECTURE_LEGACY.md`；`docs/ARCHITECTURE.md` 自本修订起只描述当前架构。开发前必读文件路径不变，仍读取 `docs/ARCHITECTURE.md`。
- 批准依据：用户要求归档旧版架构、重写当前架构，并保留带状态标签的生产升级文档。

后续修订必须从 R-003 开始追加，不得修改或删除 R-001、R-002。
