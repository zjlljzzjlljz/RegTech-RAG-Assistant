# RegTech 3.0 Agentic Compliance Assistant — Design Specification

> Date: 2026-06-25  
> Status: Approved design, ready for implementation planning  
> Target architecture: Option B — production-style modular monolith with a single Streamlit entrypoint and pluggable inference clients

---

## 1. Goal and Scope

This specification defines the complete design for upgrading the current `RegTech-RAG-Assistant` proof of concept into a production-style, modular, high-recall, agentic compliance assistant.

The target system introduces:

- Milvus as the vector database
- `BAAI/bge-m3` for hybrid dense and sparse retrieval
- multi-query expansion and HyDE query enrichment
- Reciprocal Rank Fusion (RRF) across dense and sparse candidate sets
- `BAAI/bge-reranker-large` for final reranking
- a LangGraph action layer for transaction audit workflows
- a mock SQLite transaction database and SAR generation tool
- a single Streamlit entrypoint with clear module boundaries

This design intentionally optimizes for:

- clean modular Python architecture
- strong runtime safety and graceful degradation
- production-grade boundaries and data contracts
- future portability to external inference services without requiring immediate service decomposition

This is a design document only. Code implementation begins in the next phase.

---

## 2. Chosen Architecture

### 2.1 Selected Approach

The chosen implementation model is:

**Production-style modular monolith with pluggable inference clients**

This means:

- `app.py` remains the only user-facing runtime entrypoint
- business logic is separated into focused modules under `src/`
- retrieval, indexing, tools, graph orchestration, and data access are isolated behind explicit interfaces
- LLM, embedding, reranker, and vector-store interactions are accessed through adapter-style boundaries
- the first implementation remains locally runnable, while preserving a clean upgrade path toward external inference services

### 2.2 Why this approach was selected

This design best satisfies the project constraints:

- it preserves developer ergonomics and local reproducibility
- it is significantly cleaner than the current script-evolution layout
- it avoids premature microservice overhead
- it provides production-quality separation of concerns
- it preserves a future migration path to dedicated model-serving infrastructure

---

## 3. Target 3.0 Directory Structure

```text
RegTech-RAG-Assistant/
│
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw_pdfs/
│   └── mock_db.sql
│
├── src/
│   ├── __init__.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── transaction_db.py
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── milvus_ingest.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── query_pipeline.py
│   │
│   └── agent/
│       ├── __init__.py
│       ├── tools.py
│       └── graph_agent.py
│
└── app.py
```

### 3.1 Module responsibilities

#### `config/settings.py`
- centralizes runtime configuration
- loads Milvus, SQLite, model, and API settings
- exposes a single `get_settings()` entrypoint
- contains no business logic

#### `src/database/transaction_db.py`
- initializes the mock SQLite transaction database
- seeds deterministic demo data
- exposes structured transaction query APIs
- does not depend on Streamlit, LangGraph, or Milvus

#### `src/indexing/milvus_ingest.py`
- parses PDF documents
- performs parent-child chunking
- generates BGE-M3 dense and sparse embeddings
- creates and populates Milvus collections and indexes
- handles indexing only, not query orchestration

#### `src/retrieval/query_pipeline.py`
- performs multi-query expansion
- generates HyDE hypothetical compliance text
- executes dense and sparse hybrid retrieval
- fuses results with RRF
- reranks fused candidates with a cross-encoder
- exposes one clean retrieval interface to the rest of the system

#### `src/agent/tools.py`
- provides the tool gateway used by the agent graph
- exposes database query tooling and SAR generation tooling
- avoids UI or retrieval logic leakage

#### `src/agent/graph_agent.py`
- defines the LangGraph workflow
- performs intent routing
- orchestrates transaction audit and compliance QA paths
- manages Draftee–Auditor review loops
- materializes a final UI-friendly result object

#### `app.py`
- handles the Streamlit user interface
- gathers user input and mode selection
- invokes the agent graph or retrieval path
- renders evidence, audit outcomes, and SAR output
- remains intentionally lightweight

---

## 4. High-Level Runtime Architecture

```text
User
  -> Streamlit app
  -> ComplianceAgentGraph
     -> detect intent
     -> compliance QA path OR transaction audit path
     -> retrieval pipeline
        -> multi-query expansion
        -> HyDE generation
        -> dense search
        -> sparse search
        -> RRF
        -> cross-encoder rerank
     -> Draftee analysis
     -> Auditor review loop
     -> optional SAR generation
  -> final AuditResult
  -> Streamlit rendering
```

### 4.1 Architectural principles

- UI does not know retrieval internals
- LangGraph does not know Milvus schema details
- tools do not know Streamlit session state
- database access does not know agent orchestration
- retrieval is explainable through layered scores and source tags
- no component silently converts uncertainty into a clean conclusion

---

## 5. Core Data Schemas and State Contracts

### 5.1 Shared literals

```python
RetrievalSource = Literal["dense", "sparse", "multi_query", "hyde"]
AuditIntent = Literal["compliance_qa", "transaction_audit"]
AuditDecision = Literal[
    "compliance_answer",
    "no_violation",
    "violation_detected",
    "needs_human_review",
    "failed",
]
```

### 5.2 Transaction schema

```python
@dataclass(slots=True)
class TransactionRecord:
    client_name: str
    transaction_date: str
    amount_hkd: float
    beneficiary_country: str
    status: str
```

### 5.3 Retrieval chunk schema

```python
@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    parent_id: str | None
    text: str
    source_file: str
    page_number: int | None
    dense_score: float | None
    sparse_score: float | None
    rrf_score: float
    rerank_score: float | None
    retrieval_sources: list[RetrievalSource]
    metadata: dict[str, Any] = field(default_factory=dict)
```

This schema preserves all essential evidence metadata:

- chunk text
- file name
- page number
- dense similarity score
- sparse similarity score
- RRF fusion score
- reranker score
- retrieval provenance tags

### 5.4 Audit feedback schema

```python
@dataclass(slots=True)
class AuditFeedback:
    iteration: int
    verdict: Literal["approve", "revise", "escalate"]
    summary: str
    violation_detected: bool | None
    violation_details: str | None
    recommended_changes: list[str] = field(default_factory=list)
```

### 5.5 Retrieval request schema

```python
@dataclass(slots=True)
class RetrievalRequest:
    query: str
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_top_k: int = 20
    rerank_top_k: int = 5
    enable_multi_query: bool = True
    enable_hyde: bool = True
    filters: dict[str, str | int | float] | None = None
```

### 5.6 Transaction query schema

```python
@dataclass(slots=True)
class TransactionQuery:
    client_name: str
    start_date: str | None = None
    end_date: str | None = None
    status: str | None = None
    min_amount_hkd: float | None = None
    limit: int = 100
```

### 5.7 Retrieval result schema

```python
@dataclass(slots=True)
class RetrievalResult:
    original_query: str
    expanded_queries: list[str]
    hyde_hypothesis: str | None
    retrieval_mode: Literal["hybrid", "hyde_augmented"]
    rrf_candidates: list[RetrievedChunk]
    top_chunks: list[RetrievedChunk]
    dense_candidate_count: int
    sparse_candidate_count: int
    timings_ms: dict[str, int] = field(default_factory=dict)
```

#### Retrieval invariants

- `len(top_chunks) <= 5`
- `len(rrf_candidates) <= 20`
- `top_chunks` must be a subset of `rrf_candidates`
- ranking decisions must preserve `rrf_score` and `rerank_score`

### 5.8 LangGraph state contract

```python
class AuditState(TypedDict, total=False):
    session_id: str
    user_query: str
    intent: AuditIntent
    client_name: str | None

    retrieval_result: RetrievalResult | None
    transactions: list[TransactionRecord]

    compliance_draft: str
    audit_feedback_history: list[AuditFeedback]
    current_iteration: int
    max_iterations: int

    violation_detected: bool | None
    violation_details: str | None

    sar_draft_markdown: str | None
    final_report: str | None
    final_result: "AuditResult | None"

    tool_execution_log: list[str]
    error_message: str | None
```

#### Required semantic meaning

- `user_query` stores the original user request
- `compliance_draft` stores the current Draftee output
- `audit_feedback_history` stores append-only Auditor history
- `current_iteration` tracks completed audit rounds
- `final_report` stores the final formatted audit conclusion

#### State invariants

- `audit_feedback_history` is append-only
- `current_iteration` increments only after `audit_analysis_node`
- `final_report` becomes immutable once finalized
- `final_result` is written only in the output node

### 5.9 Final output schema

```python
@dataclass(slots=True)
class AuditResult:
    session_id: str
    intent: AuditIntent
    decision: AuditDecision

    response_markdown: str
    final_report: str

    violation_detected: bool | None
    violation_details: str | None

    audited_transactions: list[TransactionRecord]
    cited_chunks: list[RetrievedChunk]

    sar_draft_markdown: str | None
    sar_report_path: str | None

    total_iterations: int
    tool_execution_log: list[str] = field(default_factory=list)
```

#### Output invariants

- UI consumes only `AuditResult`
- `cited_chunks` must come from `RetrievalResult.top_chunks`
- `audited_transactions` must be equal to or a subset of state transactions
- a dependency failure must never produce `no_violation`

---

## 6. Interface Design

### 6.1 Database interface

The database layer exposes a repository abstraction.

```python
class TransactionRepository:
    def __init__(self, db_path: str) -> None:
        pass

    def initialize(self) -> None:
        pass

    def seed_demo_data(self, force_reset: bool = False) -> int:
        pass

    def query_transactions(self, query: TransactionQuery) -> list[TransactionRecord]:
        pass

    def query_client_transactions(self, client_name: str) -> list[TransactionRecord]:
        pass

    def list_clients(self) -> list[str]:
        pass
```

#### Database responsibilities

- initialize schema and indexes
- seed mock data deterministically
- support structured transaction lookups
- return typed records only
- avoid presentation formatting and orchestration logic

### 6.2 Retrieval interface

The retrieval layer exposes one high-level entrypoint.

```python
class ComplianceRetrievalPipeline:
    def __init__(
        self,
        embedding_client: "EmbeddingClient",
        reranker_client: "RerankerClient",
        vector_store: "MilvusHybridStore",
        llm_provider: "LLMProvider",
    ) -> None:
        pass

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        pass
```

#### Suggested internal methods

```python
class ComplianceRetrievalPipeline:
    def expand_queries(self, query: str, n: int = 3) -> list[str]:
        pass

    def generate_hyde(self, query: str) -> str | None:
        pass

    def dense_search(self, query_texts: list[str], top_k: int = 50) -> list[RetrievedChunk]:
        pass

    def sparse_search(self, query_texts: list[str], top_k: int = 50) -> list[RetrievedChunk]:
        pass

    def reciprocal_rank_fusion(
        self,
        dense_hits: list[RetrievedChunk],
        sparse_hits: list[RetrievedChunk],
        top_k: int = 20,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        pass

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        pass
```

#### Retrieval orchestration order

1. expand the user query into three professional regulatory variants
2. optionally generate a HyDE hypothetical compliance answer
3. run dense retrieval over the query set
4. run sparse retrieval over the query set
5. fuse candidates with RRF
6. rerank the fused top-20 to final top-5
7. return a fully populated `RetrievalResult`

### 6.3 Tool gateway interface

```python
def query_client_transactions(client_name: str) -> list[dict]:
    pass


def generate_compliance_report(client_name: str, violation_details: str) -> str:
    pass


def build_agent_tools(
    repository: TransactionRepository,
    report_output_dir: str,
) -> list[object]:
    pass
```

#### Tool gateway responsibilities

- wrap database and reporting operations for agent consumption
- return deterministic, minimally formatted outputs
- avoid embedding retrieval logic or UI behavior

---

## 7. LangGraph Node and State Flow Design

### 7.1 Nodes

The graph contains seven nodes.

1. `detect_intent_node`
2. `load_transactions_node`
3. `retrieve_evidence_node`
4. `draft_analysis_node`
5. `audit_analysis_node`
6. `generate_sar_node`
7. `finalize_output_node`

### 7.2 Node definitions

#### `detect_intent_node`

Inputs:
- `user_query`

Writes:
- `intent`
- `client_name`
- `tool_execution_log`

Behavior:
- classifies whether the request is `compliance_qa` or `transaction_audit`
- extracts `client_name` when possible
- falls back to lightweight keyword rules if LLM classification fails

#### `load_transactions_node`

Inputs:
- `client_name`

Writes:
- `transactions`
- `tool_execution_log`
- optionally `error_message`

Behavior:
- retrieves transaction records from SQLite
- does not interpret results
- missing results produce a controlled no-data outcome, not a clean audit conclusion

#### `retrieve_evidence_node`

Inputs:
- `user_query`
- `transactions` when in audit mode

Writes:
- `retrieval_result`
- `tool_execution_log`
- optionally `error_message`

Behavior:
- constructs the retrieval request
- runs hybrid retrieval against HKMA and related guidance sources
- may use a transaction summary to strengthen the audit retrieval query

#### `draft_analysis_node`

Inputs:
- `intent`
- `user_query`
- `transactions`
- `retrieval_result`
- `audit_feedback_history`

Writes:
- `compliance_draft`
- `tool_execution_log`
- optionally `error_message`

Behavior:
- generates the current answer or audit draft
- incorporates prior Auditor feedback when present
- remains fully evidence grounded

#### `audit_analysis_node`

Inputs:
- `compliance_draft`
- `retrieval_result`
- `transactions`
- `intent`
- `current_iteration`
- `max_iterations`

Writes:
- `audit_feedback_history`
- `current_iteration`
- `violation_detected`
- `violation_details`
- `tool_execution_log`
- optionally `error_message`

Behavior:
- reviews groundedness, regulatory precision, completeness, and audit reasoning quality
- appends a new `AuditFeedback`
- increments the iteration counter after each completed audit round

#### `generate_sar_node`

Inputs:
- `client_name`
- `violation_details`

Writes:
- `sar_draft_markdown`
- `tool_execution_log`
- optionally `error_message`

Behavior:
- generates a deterministic SAR draft only when a violation is confirmed
- does not reopen the Draftee–Auditor loop

#### `finalize_output_node`

Inputs:
- entire graph state

Writes:
- `final_report`
- `final_result`
- `tool_execution_log`

Behavior:
- converts internal state into a stable `AuditResult`
- determines the final decision classification
- returns a UI-ready structure

### 7.3 Conditional routing functions

#### `route_after_intent(state: AuditState) -> str`

Rules:
- `intent == "compliance_qa"` -> `retrieve_evidence_node`
- `intent == "transaction_audit"` and `client_name` exists -> `load_transactions_node`
- `intent == "transaction_audit"` and `client_name` missing -> `finalize_output_node`

Reasoning:
- audit requests must identify a concrete audit subject before transaction analysis begins

#### `route_after_transactions(state: AuditState) -> str`

Rules:
- non-empty `transactions` -> `retrieve_evidence_node`
- empty `transactions` -> `finalize_output_node`

Reasoning:
- missing transaction data is a business-data insufficiency and must not be treated as a clean audit pass

#### `route_after_audit(state: AuditState) -> str`

Rules:
- latest verdict is `approve` and `intent == "compliance_qa"` -> `finalize_output_node`
- latest verdict is `approve` and `violation_detected is True` -> `generate_sar_node`
- latest verdict is `approve` and `violation_detected is False` -> `finalize_output_node`
- latest verdict is `revise` and `current_iteration < max_iterations` -> `draft_analysis_node`
- latest verdict is `revise` and `current_iteration >= max_iterations` -> `finalize_output_node`
- latest verdict is `escalate` -> `finalize_output_node`

#### `generate_sar_node` outbound flow

- always routes to `finalize_output_node`

### 7.4 Iteration semantics

- `current_iteration` starts at `0`
- `audit_analysis_node` increments it after each completed audit review
- with `max_iterations = 3`, the system allows at most three Auditor passes
- the graph never loops indefinitely

### 7.5 Full state flow

```text
START
  -> detect_intent_node

detect_intent_node
  -> retrieve_evidence_node           if intent == compliance_qa
  -> load_transactions_node           if intent == transaction_audit and client_name exists
  -> finalize_output_node             if transaction_audit but client_name missing

load_transactions_node
  -> retrieve_evidence_node           if transactions found
  -> finalize_output_node             if no transactions found

retrieve_evidence_node
  -> draft_analysis_node

draft_analysis_node
  -> audit_analysis_node

audit_analysis_node
  -> draft_analysis_node              if verdict == revise and current_iteration < max_iterations
  -> generate_sar_node                if verdict == approve and violation_detected == True
  -> finalize_output_node             if verdict == approve and violation_detected == False
  -> finalize_output_node             if verdict == approve and intent == compliance_qa
  -> finalize_output_node             if verdict == escalate
  -> finalize_output_node             if verdict == revise and current_iteration >= max_iterations

generate_sar_node
  -> finalize_output_node

finalize_output_node
  -> END
```

### 7.6 Main runtime paths

#### Compliance QA path

- user asks a regulatory question
- graph classifies `compliance_qa`
- evidence is retrieved from the guidance corpus
- Draftee produces an evidence-grounded answer
- Auditor checks groundedness and completeness
- final result is returned with `decision = "compliance_answer"`

#### Transaction audit path

- user asks to audit a named client’s transactions
- graph classifies `transaction_audit`
- repository returns transaction records
- retrieval pipeline finds relevant AML/CFT guidance
- Draftee compares facts against policy requirements
- Auditor decides whether risk is supported, unsupported, or inconclusive
- if a violation is confirmed, SAR generation runs
- final result becomes `no_violation`, `violation_detected`, or `needs_human_review`

---

## 8. Error Handling Design

### 8.1 Governing safety rule

**When uncertain, escalate. When unavailable, degrade. When incomplete, never issue a clean no-violation conclusion.**

This rule is mandatory for all audit-related paths.

### 8.2 Error taxonomy

#### `ConfigurationError`
- missing API key
- invalid model name
- broken Milvus host or collection configuration
- startup-time misconfiguration

#### `DependencyUnavailableError`
- Milvus connection loss
- inference client unavailability
- file-system write failure

#### `RateLimitError`
- provider 429 responses
- temporary quota exhaustion

#### `DataAccessError`
- SQLite schema mismatch
- transaction query failure
- corrupted local database

#### `ValidationError`
- malformed structured LLM output
- missing required tool arguments
- invalid state transitions

### 8.3 Node-specific graceful degradation

#### Intent detection
- primary strategy: LLM classification
- fallback strategy: lightweight keyword-based classification
- final safe default: `compliance_qa`

#### Transaction loading
- SQLite exception -> write `error_message`, append log event, route to `finalize_output_node`
- empty transaction set -> no-data outcome, not a clean pass
- final output should be `needs_human_review` rather than `no_violation`

#### Retrieval
- Milvus unavailable during `compliance_qa` -> return `failed`
- Milvus unavailable during `transaction_audit` -> return `needs_human_review`
- dense failure + sparse success -> sparse-only fallback
- sparse failure + dense success -> dense-only fallback
- reranker failure -> use RRF top-5 directly
- HyDE failure -> disable HyDE and continue main retrieval chain

#### Drafting and auditing
- rate limit or timeout -> exponential backoff retries at increasing intervals, max 3 attempts
- repeated failure in `compliance_qa` -> `failed`
- repeated failure in `transaction_audit` -> `needs_human_review`
- invalid structured output -> one repair attempt, then controlled stop

#### SAR generation
- SAR generation failure does not erase a confirmed violation
- final decision remains `violation_detected`
- `sar_draft_markdown` may be empty with an explicit generation failure note

### 8.4 Graph-level containment strategy

Every node must follow the same safety wrapper pattern:

1. catch internal exceptions
2. write `error_message`
3. append a structured event to `tool_execution_log`
4. route to a safe finalization path

Unhandled exceptions must not escape the graph directly.

### 8.5 Safety decision matrix

| Failure mode | compliance_qa result | transaction_audit result |
|---|---|---|
| Milvus unavailable | `failed` | `needs_human_review` |
| SQLite failure | n/a | `needs_human_review` |
| Claude temporary rate limit | retry | retry |
| Claude repeated failure | `failed` | `needs_human_review` or `failed` |
| SAR generation failure after confirmed violation | n/a | `violation_detected` |

---

## 9. Observability Design

### 9.1 Logging goals

The system does not require commercial APM tooling in the first implementation. Instead, it uses structured Python logging plus an append-only in-state execution log.

Every important event should include:

- `session_id`
- `node_name`
- `intent`
- `client_name` when present
- `latency_ms`
- `status`
- `error_type` when applicable

### 9.2 Logging levels

#### `INFO`
- node start and completion
- retrieval candidate counts
- auditor verdicts
- final decision classification

#### `WARNING`
- graceful degradation events
- reranker fallback usage
- HyDE failure
- one-sided retrieval fallback

#### `ERROR`
- repeated provider failure
- Milvus unavailability
- SQLite access failure
- unrecoverable state validation failure

#### `DEBUG`
- prompt lengths
- candidate chunk IDs
- raw RRF ordering details
- intermediate retrieval metrics

### 9.3 Metrics to record

#### Retrieval stage timings
- query expansion latency
- HyDE latency
- dense search latency
- sparse search latency
- RRF latency
- rerank latency

#### LLM usage metrics
- prompt tokens
- completion tokens
- total tokens

#### End-to-end metrics
- total graph runtime
- per-node latency
- total audit iterations

### 9.4 `tool_execution_log` format

The graph state preserves an append-only `list[str]` with stable machine-readable messages.

Examples:

```text
[2026-06-25T10:21:18Z] node=retrieve_evidence event=completed latency_ms=842 dense=50 sparse=50 rrf=20 rerank=5
[2026-06-25T10:21:21Z] node=draft_analysis event=completed latency_ms=1390 prompt_tokens=812 completion_tokens=233 total_tokens=1045
[2026-06-25T10:21:24Z] node=audit_analysis event=warning reason=rate_limit retry=2 latency_ms=4120
```

### 9.5 Token accounting through provider adapters

The LLM provider adapter is responsible for extracting token usage from model responses and normalizing it into:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

These values are recorded both in:

- Python logs
- `tool_execution_log`

### 9.6 Latency accounting

Every node should use high-resolution timing and write:

- node runtime
- retrieval-stage timings
- total workflow runtime

The final output node is responsible for surfacing the complete end-to-end timing summary.

---

## 10. Testing Strategy

### 10.1 Testing principles

The system must be testable without requiring real Milvus or live LLM calls in default CI.

The core strategy is:

- deterministic unit tests for local logic
- integration tests with fake or stubbed dependencies
- optional external tests for real infra validation

### 10.2 Suggested test structure

```text
tests/
├── unit/
│   ├── database/
│   │   └── test_transaction_db.py
│   ├── retrieval/
│   │   ├── test_rrf.py
│   │   └── test_query_pipeline_degradation.py
│   └── agent/
│       └── test_routing.py
└── integration/
    ├── test_graph_compliance_qa.py
    └── test_graph_transaction_audit.py
```

### 10.3 Unit tests for `TransactionRepository`

Cover:

- schema creation succeeds
- demo seeding inserts expected record counts
- `query_client_transactions()` returns expected rows
- `query_transactions()` applies date, status, and amount filters correctly
- empty results return empty lists, not crashes
- malformed database access raises controlled exceptions

### 10.4 Unit tests for retrieval

Cover:

- RRF merges dense and sparse rankings correctly
- duplicate chunk IDs merge correctly
- fallback behavior works when one retrieval path fails
- reranker fallback uses RRF top-k when reranking is unavailable
- `RetrievalResult` contract is always valid

### 10.5 Unit tests for graph routing

Cover:

- `compliance_qa` routes directly to retrieval
- `transaction_audit` with client name routes to transaction loading
- `transaction_audit` without client name finalizes early
- max-iteration protection stops loops correctly
- confirmed violations route to SAR generation
- escalation routes finalize safely

### 10.6 Integration tests for the graph

Use dependency injection with fake providers for:

- transaction repository
- retrieval pipeline
- LLM provider
- SAR generator

Scenarios to cover:

1. compliance QA success path
2. transaction audit with confirmed violation and SAR generation
3. transaction audit with no violation
4. repeated revise verdicts until `max_iterations`
5. Milvus retrieval failure causing safe downgrade
6. SQLite or LLM failure producing `needs_human_review` or `failed`

### 10.7 Assertions that must always hold

- `intent` routing is correct
- `current_iteration` increments correctly and never exceeds policy behavior
- a confirmed violation triggers SAR generation
- dependency failure never yields `no_violation`
- `AuditResult.cited_chunks` come from `RetrievalResult.top_chunks`

### 10.8 External test policy

Optional real-environment tests may be marked separately:

- `@pytest.mark.external`
- `@pytest.mark.milvus`
- `@pytest.mark.anthropic`

Default CI should run only deterministic unit and integration tests with fake dependencies.

---

## 11. Implementation Guidance Summary

The next implementation phase should generate code that satisfies the following non-negotiable constraints:

- use standard module imports, not dynamic script loading
- keep `app.py` thin and orchestration-focused
- centralize retrieval complexity in `query_pipeline.py`
- preserve explainability through layered retrieval scores
- preserve safety through explicit final decision categories
- never issue a clean no-violation conclusion when key dependencies are unavailable
- support future provider replacement through adapter-style interfaces

---

## 12. Final Readiness Statement

This design is complete and internally consistent.

It defines:

- the target module structure
- the runtime architecture
- the key data schemas and state contracts
- the retrieval and database interface boundaries
- the LangGraph nodes and conditional flow
- the iteration guard behavior
- the error-handling and graceful-degradation model
- the observability strategy
- the unit and integration test plan

The design is ready for implementation planning and then full code generation.
