# RegTech RAG Assistant — 工程架构技术方案

> **历史归档，不再维护。** 本文记录 ChromaDB、MiniLM、Claude 和早期编号脚本阶段的原型设计，仅用于追溯架构演进。当前架构请阅读 [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)，开发规范以 [`../../../PROJECT_RULES.md`](../../../PROJECT_RULES.md) 为准。

> 当前生产化架构、模型服务、v2 索引迁移和评估门禁以 [`../../PRODUCTION_UPGRADE.md`](../../PRODUCTION_UPGRADE.md) 为准。本文件后续章节保留早期原型设计背景。

> 面试用技术说明书 | 面向 HKMA 合规场景的隐私优先 RAG 系统

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构全景图](#2-架构全景图)
3. [五大关键设计决策](#3-五大关键设计决策)
4. [技术栈与数据流](#4-技术栈与数据流)
5. [项目文件结构](#5-项目文件结构)
6. [效果评估](#6-效果评估)
7. [演进路线](#7-演进路线)
8. [面试问答要点](#8-面试问答要点)

---

## 1. 项目概述

一个面向 **HKMA（香港金融管理局）AML/CFT 合规场景** 的 **隐私优先 RAG（检索增强生成）系统**，从单路向量检索演进为 **自适应混合检索 + Cross-Encoder 重排 + LangGraph 双 Agent 审计闭环**。

### 核心特性

```
输入：用户用自然语言提问合规问题
输出：带逐条页码引用的审计级合规报告
约束：所有文档和 embedding 永不出本地
```

### 量化指标

| 指标 | 数值 |
|------|------|
| 知识库规模 | 91 parent + 263 child 节点 |
| 检索通道 | Fast Path (< 3s) / Rescue Path (~10s) |
| 审计闭环 | Draftee→Auditor→Feedback，最多 3 轮迭代 |
| 隐私合规 | 全链路零外部 API 调用（除最后 Claude 生成环节） |
| 降级鲁棒性 | 4-tier 模型降级 + 条件 HyDE 触发 |

---

## 2. 架构全景图

```
                        用户提问
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │       Adaptive Retrieval Router       │
        │                                        │
        │  Fast Path            Rescue Path      │
        │  Dense → Cross-Encoder  HyDE → Dense  │
        │  (< 3s)               → Cross-Encoder  │
        │                       (~10s)           │
        └──────────┬───────────────┬─────────────┘
                   │               │
                   ▼               ▼
        ┌──────────────────────────────────────┐
        │   ChromaDB (Parent-Child Index)       │
        │   91 parent nodes / 263 child nodes   │
        │   Local HuggingFace embeddings (CPU)  │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │         Claude (via API)              │
        │   4-tier graceful degradation         │
        │   temperature = 0 (deterministic)     │
        └──────────────────┬───────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌─────────────────┐     ┌─────────────────────┐
    │  Single-Pass RAG │     │  Dual-Agent Audit    │
    │  (Strict/BG mode)│     │  (Audited Report)    │
    │                  │     │                      │
    │  检索 → 生成 → 输出│     │  Draftee → Auditor   │
    │                  │     │     ↑        │        │
    │                  │     │     └──REJECTED──┘    │
    │                  │     │  (feedback loop × 3)  │
    └─────────────────┘     └─────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   Streamlit UI       │
                          │   Citations + Audit  │
                          │   Trail + Sidebar    │
                          └─────────────────────┘
```

---

## 3. 五大关键设计决策

### 决策 1：本地 Embedding 而非云 API

| 方案 | 精度 | 隐私 | 成本 |
|------|------|------|------|
| OpenAI text-embedding-3-large | ⭐⭐⭐ | ❌ 数据上传外部 | 按量计费 |
| Cohere Embed v3 | ⭐⭐⭐ | ❌ 同上 | 按量计费 |
| **all-MiniLM-L6-v2 本地 CPU** | ⭐⭐ | ✅ 零外部调用 | 免费 |

**选择原因：** 银行合规部门的数据安全红线——敏感监管文档不得离开内网。`sentence-transformers` 在 CPU 上运行，384 维向量对监管文本的语义区分能力经实测验证足够。

**面试话术：** "合规部门的硬约束是数据不得离开内网。我用 sentence-transformers 本地运行 embedding，ChromaDB 本地持久化，整个检索链路零外部 API 调用。只有最后的生成环节调用 Claude。"

---

### 决策 2：Parent-Child 分层索引而非 Flat Chunk

**问题：** 等长切块（800 字符 + 150 重叠）会把 `Section 4.1(a)(iii)` 和其上下文拦腰截断。检索命中小碎片，Claude 看到的是一段被割裂的文字。

**方案：**

```
LlamaIndex HierarchicalNodeParser:

  原始 PDF 页面
       │
       ▼
  Parent Node (1024 tokens)
  = 完整段落 / 法规条款
       │
       ├── Child Node (256 tokens)
       ├── Child Node (256 tokens)    ← 索引用细粒度碎片
       └── Child Node (256 tokens)

  检索流程:
    查询 → 搜 Child（精准匹配、召回率高）
         → 找到 Child #5
         → 返回 Parent #3（完整上下文、Claude 能看懂）
```

**实测效果：** 原版 flat chunk 的检索结果常出现同一页的多个碎片（浪费槽位），parent-child 确保了每条结果都是完整段落。

**面试话术：** "监管文档是层级结构的——一条法规有一级条款、二级细则。等长切块会破坏这种结构。用 LlamaIndex 的 HierarchicalNodeParser 建立父子节点关系，检索小粒度 child 保证召回精度，返回大粒度 parent 保证上下文完整。"

---

### 决策 3：Cross-Encoder 重排而非纯向量检索

**问题：** Bi-Encoder（双塔模型）把查询和文档分别编码成向量再算余弦距离——两个向量从未在模型层内"见面"。封面页上的 "Guideline on Anti-Money Laundering" 和正文 Chapter 4 里的 "customer due diligence measures for AML" 在向量空间里可能很近，但封面是废信息。

**方案：**

```
第一轮 Bi-Encoder（召回）:
  100 份文档 → 分别编码 → 余弦距离 → Top 10
  特点: 快（O(n) 向量计算）、粗（无交叉注意力）

第二轮 Cross-Encoder（精排）:
  Top 10 → 逐对 (查询, 文档) 拼接输入 Transformer
         → 交叉注意力在所有层交互 → 相关性分数
  特点: 慢（每对单独推理）、准（查询和文档在模型中"对话"）

最终: Top 3 精排 → 喂给 Claude
```

**实测效果：** 对比测试中，Cross-Encoder 成功将封面页和术语表从高排位踢到低排位，把 Chapter 3 AML/CFT Systems 从 dense #4 提到 reranked #1。

**面试话术：** "Bi-Encoder 把查询和文档分别编码成向量再算余弦距离——两个向量从未'见面'。Cross-Encoder 把查询和文档拼接输入，在 Transformer 每一层交叉注意力，能捕捉'risk assessment procedures'和'Risk-Based Approach'是同一个意思。代价是每对要单独推理一次，所以只对 Top 10 做。"

---

### 决策 4：条件触发 HyDE 而非无脑展开

**问题：** 用户说 "risk assessment procedures for money laundering"，文档里写 "Risk-Based Approach"、"institutional ML/TF risk assessment"。语义鸿沟导致向量检索失败——两者在 embedding 空间中距离偏远。

**方案（HyDE = Hypothetical Document Embeddings）：**

```
常规路径:
  用户查询 → embedding → 检索 → 结果（可能失败）

HyDE 路径:
  用户查询 → Claude 生成假设性法规段落 → embedding → 检索 → 结果

  生成的段落（示例）:
  "An authorized institution should establish and maintain documented
   ML/TF risk assessment procedures under a Risk-Based Approach,
   taking into account customer, product, delivery channel, transaction,
   and geographic risk factors..."
  → 这段与 Chapter 2 "Risk-Based Approach" 高度同频 → 检索成功
```

**关键工程判断：不要对所有查询都用 HyDE。**

```
条件触发逻辑:
  Cross-Encoder 检索 → 如果 top-1 得分 ≥ 0
    → Fast Path（< 3s，直接返回）
  Cross-Encoder 检索 → 如果 top-1 得分 < 0
    → Rescue Path（~10s，Claude 展开查询 → 重新检索）
```

**实测效果：** Q3 "risk assessment procedures" 是三版检索器中唯一能命中 Chapter 2 Risk-Based Approach 的方案。

**面试话术：** "这是整个系统最体现工程判断力的设计。HyDE 本身不新鲜，但我没有无脑给所有查询都加——因为每次调 Claude 展开查询要多 6-7 秒延迟。我做了条件触发：先用 Cross-Encoder 快速路径，如果 top-1 得分 < 0，说明当前检索质量不够，才触发 HyDE 救援路径。这叫 Adaptive HyDE——快查询走快速路，难查询自动救援。"

---

### 决策 5：双 Agent 审计闭环而非单轮 RAG

**问题：** 合规场景下，幻觉不是"出错"是"违规"。单程 RAG（检索 → 生成 → 输出）无机制保证 Claude 真正引用了检索到的法规原文，也无机制验证引用的准确性。

**方案：**

```
              ┌──────────┐
              │ Retrieve │  ← Cross-Encoder 检索证据
              └────┬─────┘
                   │
                   ▼
              ┌──────────┐
         ┌───→│ Draftee  │  ← 基于证据起草合规报告
         │    │ (起草官) │     "每条事实必须带 [Source: file, Page: X]"
         │    └────┬─────┘
         │         │
         │         ▼
         │    ┌──────────┐
         │    │ Auditor  │  ← 逐条审查草稿
         │    │ (审查官) │     ① 每条事实有源文件+页码?
         │    └────┬─────┘     ② 是否引用了证据之外的法规?
         │         │           ③ 是否遗漏了证据中的重要内容?
         │    ┌────┴────┐
         │    │ APPROVED?│
         │    ├─ YES → END ──→ 输出审计合规报告
         │    └─ NO ─→ audit_feedback
         │              │
         └──────────────┘  (最多 3 轮)
```

**LangGraph 状态机实现：**

- `State`: `{messages, evidence, compliance_draft, audit_feedback, audit_approved, iteration_count}`
- `retrieve_node`: 调用 Cross-Encoder 检索
- `draftee_node`: 基于证据 + 审计反馈（如有）生成/修改草稿
- `auditor_node`: 审查草稿，输出 APPROVED 或 REJECTED + 具体反馈
- `Conditional Edge`: `audit_approved OR iteration >= 3 → END; else → draftee_node`

**实测效果：** 6 条测试查询全部执行完整审计循环。Auditor 能指出草稿中具体缺失的页码和内容（如"缺少 p.47 非面对面措施"、"缺少 p.22 法人识别要求"），Draftee 逐轮补充完善。

**面试话术：** "合规场景下，幻觉不是'出错'而是'违规'。单程 RAG 无法保证 Claude 一定引用了检索到的法规原文。双 Agent 架构用 LangGraph 状态机实现审计闭环：起草官写报告，审查官逐条核验——每条事实陈述是否带页码？是否引用了检索证据？源文件是否匹配？不通过就退回重写，最多 3 轮。这是把合规审计的'四眼原则'编码进了 AI pipeline。"

---

## 4. 技术栈与数据流

### 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| UI | Streamlit 1.50 | 对话界面、侧边栏、审计面板 |
| 状态机 | LangGraph 0.6 | 双 Agent 审计闭环 |
| 生成 | Anthropic Claude SDK 0.109 | 合规报告生成 + 审计 |
| 检索 | ChromaDB 1.5 + Sentence-Transformers | 向量存储 + 本地 embedding |
| 重排 | Cross-Encoder (ms-marco-MiniLM-L-6-v2) | 检索结果精排 |
| 文档解析 | LlamaIndex + PyMuPDF | PDF 加载 + 层级切块 |
| 查询展开 | HyDE (Claude 生成) | 模糊查询 → 假设性段落 |
| 依赖管理 | pip + requirements.txt | 可复现环境 |
| 代码组织 | importlib 模块化导入 | 每个检索器独立文件/独立测试 |

### 完整数据流

```
1. 用户输入问题
        │
2. retrieve_evidence_adaptive()
   ├── Fast Path: Cross-Encoder 检索 top-5
   │   └── 如果 top-1 score ≥ 0 → 返回结果
   └── Rescue Path: HyDE 查询展开 → 重新检索 → 返回结果
        │
3. build_api_messages()
   ├── 拼接最近 12 轮对话历史（纯文本）
   ├── 拼接检索到的证据块 (E1, E2, E3...)
   └── 拼接当前回答模式 + 用户问题
        │
4. stream_claude_answer()
   ├── Tier 1: Claude-fable-5 + adaptive thinking + output_config
   ├── Tier 2: Claude-fable-5 + temperature=0（降级）
   ├── Tier 3: fallback 模型（备选）
   └── Tier 4: 最终 fallback
        │
5. [审计模式] run_audited_compliance_report()
   ├── Draftee: 生成合规报告草稿
   ├── Auditor: 审查引用完整性
   ├── 条件循环: 最多 3 轮反馈修改
   └── 输出最终审计报告
        │
6. select_citations()
   ├── 用 regex r"E\d+" 提取回复中的证据引用
   └── 映射回源文件 + 页码
        │
7. Streamlit UI 渲染
   ├── 聊天历史（含引用标注）
   ├── 审计轨迹（含 audit trail expander）
   └── 侧边栏（检索路径、模型状态、降级记录）
```

---

## 5. 项目文件结构

```
RegTech-RAG-Assistant/
├── .gitignore                    # 排除 .env / .venv / chroma_db / *.log
├── .env                          # API Key + Base URL（不入库）
├── requirements.txt              # Python 依赖清单
├── Makefile                      # make build / test / app / app-v2 / app-v3
│
├── data/
│   └── hkma_aml_guidelines.pdf   # HKMA AML/CFT 合规指南 (91页)
│
├── chroma_db/                    # ChromaDB 持久化数据（不入库）
│
├── 1_build_db.py                 # [Phase 0] 原版 flat chunk 入库
├── 2_test_db.py                  # [Phase 0] ChromaDB 连接验证
├── 3_app.py                      # [Phase 0] 原版单路向量检索 Streamlit 应用
│
├── 4_llamaindex_ingest.py        # [Phase 1] LlamaIndex 父子文档解析
│   └── HierarchicalNodeParser: 91 parent + 263 child
│   └── 写入 ChromaDB collection: regtech_parent_child_docs
│
├── 5_hybrid_retrieval.py         # [Phase 2-实验] Dense + BM25 + RRF 混合检索
│   └── 已弃用（BM25 对监管文档引入噪声）
│
├── 6_cross_encoder_retrieval.py  # [Phase 2] Cross-Encoder 重排检索器
│   └── Dense Top 10 → CrossEncoder 精排 → Top 5
│   └── 独立可测: python 6_cross_encoder_retrieval.py
│
├── 7_hyde_retrieval.py           # [Phase 2B] HyDE 查询展开 + 三路对比基准
│   └── Dense vs X-Encoder vs HyDE+XE 对比表
│   └── 条件 Rescue Path 触发逻辑
│
├── 8_app.py                      # [Phase 3] 自适应检索 Streamlit 应用
│   └── Fast/Rescue Path + Cross-Encoder + 条件 HyDE
│   └── 端口: 8502 (make app-v2)
│
├── 9_langgraph_agent.py          # [Phase 4] LangGraph 双 Agent CLI
│   └── StateGraph: retrieve → draftee → auditor → (conditional loop)
│   └── 独立可测: python 9_langgraph_agent.py
│
├── 10_app.py                     # [Phase 5] 三模式 Streamlit 应用
│   └── Strict Grounding / Background / Audited Compliance Report
│   └── Audit trail expander + citation display
│   └── 端口: 8503 (make app-v3)
│
├── compare_retrieval.py          # 检索对比工具（新旧三路对比）
└── comparison_results.json       # 对比基准数据
```

**架构原则：** 每个文件独立可运行、独立可测试。`4_llamaindex_ingest.py` 到 `10_app.py` 通过 importlib 懒加载复用，不修改已有文件。这是微服务思想在 monolith Python 项目中的实践。

---

## 6. 效果评估

### 检索质量对比

| 查询 | Flat Dense | +Cross-Encoder | +HyDE+XE |
|------|-----------|---------------|----------|
| Q1: CDD requirements | ⭐⭐ (p.34, 脱靶) | ⭐⭐⭐ (p.19 Ch.4) | ⭐⭐ (过拟合) |
| Q2: Section 4.1 AML/CFT | ⭐ (p.3 Overview) | ⭐⭐⭐ (p.14 Ch.3) | ⭐⭐⭐ (p.14 Ch.3) |
| Q3: risk assessment procedures | ⭐ (p.5 ML stages) | ⭐ (p.90 Glossary) | ⭐⭐⭐ (p.10 Ch.2 RBA) |

### 审计闭环验证

| 查询 | 迭代次数 | 审计结果 | 关键发现 |
|------|---------|----------|----------|
| CDD measures | 3 轮 | 全部 REJECTED | Auditor 逐轮找到缺失页码(p.47 中介/p.25 身份验证)，Draftee 逐轮补充 |
| Identity verification | 3 轮 | 全部 REJECTED | Auditor 要求补充法人识别要求(p.22) |
| Risk assessment | 1 轮 | APPROVED | 证据弱但引用正确，Auditor 放行 |

### 性能指标

| 指标 | Fast Path | Rescue Path | Audit Mode |
|------|-----------|-------------|------------|
| 平均延迟 | 2-3s | 8-12s | 30-90s |
| Claude 调用次数 | 1 | 2 | 2-7 (含审计循环) |
| 检索覆盖 | top-5 | top-5 (HyDE) | top-3 |

---

## 7. 演进路线

```
Phase 0: Flat Chunk + 单路向量检索
    ↓
Phase 1: LlamaIndex Parent-Child 层级解析
    ↓
Phase 2: Cross-Encoder 重排 → 淘汰 BM25
    ↓
Phase 2B: Adaptive HyDE 条件查询展开
    ↓
Phase 3: 自适应检索 Streamlit 应用 (8_app.py)
    ↓
Phase 4: LangGraph 双 Agent CLI (9_langgraph_agent.py)
    ↓
Phase 5: 三模式 Streamlit 应用 (10_app.py)
    ↓
未来: RAGAS 量化评估 / 多文档知识库 / LangGraph 应用集成
```

---

## 8. 面试问答要点

### Q: 为什么不用 LangChain 全家桶？

> LangChain 适合快速原型，但它的高层抽象在需要精细控制检索行为（如 parent-child 回填、Cross-Encoder 重排、条件 HyDE 触发）时反而是负担。我用 importlib 模块化导入，每个检索器独立文件、独立测试、独立版本管理——这是微服务思想在 monolithic Python 项目里的投射。

### Q: 为什么不用更贵的 embedding 模型？

> 银行合规场景的硬约束是数据不出内网。all-MiniLM-L6-v2 在 CPU 上运行，384 维向量对监管文本的语义区分能力经实测验证足够。如果有 GPU，可以升级到 bge-large-en-v1.5 (1024 维)，但当前方案在精度和隐私之间取得了平衡。

### Q: HyDE 每次都调 Claude，不慢吗？

> 不无脑用。我做了条件门控：Cross-Encoder 检索后如果 top-1 相关性得分 ≥ 0，说明当前检索质量够了，直接走 Fast Path（< 3s）。只有得分 < 0 时才触发 Rescue Path 用 HyDE 展开查询。这种 Adaptive HyDE 设计保证了常见查询的响应速度，同时为困难查询提供了兜底。

### Q: 双 Agent 审计为什么比单轮 RAG 好？

> 合规场景下，幻觉不是"出错"是"违规"。单轮 RAG 无机制保证 Claude 真正引用了检索到的证据。双 Agent 审计把合规业的"四眼原则"编码进 AI 流程——起草官写报告，审查官逐条核验引用完整性。实测中，Auditor 能精确定位草稿中缺失的具体页码和内容。审计闭环确保最终输出的每一条事实都有可追溯的源文件+页码引用。

### Q: 这个系统最大的技术亮点是什么？

> 不是某个单一技术，而是**工程判断力驱动的架构演进**。从 flat chunk 到 parent-child，从纯向量到 Cross-Encoder 重排，从无脑 HyDE 到条件触发，从单轮 RAG 到双 Agent 闭环——每一步都是基于实测数据（不是理论偏好）做出的演进决策。comparison_results.json 里存了每一版的对比基准，可以回溯证明每个设计决策的价值。

---

> 生成日期: 2026-06
> 项目地址: https://github.com/zjlljzzjlljz/RegTech-RAG-Assistant
