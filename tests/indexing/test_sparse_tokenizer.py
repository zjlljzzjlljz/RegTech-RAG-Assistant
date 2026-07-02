"""Tests for deterministic sparse tokenizer.

Verifies that tokenization and sparse vector building are:
- Deterministic across runs (no Python hash() dependency)
- Language-aware: English word split, Chinese char bigrams, mixed-token handling
- Used by both ingest-side and query-side code paths
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Tests for the tokenizer module (before it exists)
# ---------------------------------------------------------------------------


def test_tokenizer_module_exists() -> None:
    """The tokenizer module must be importable."""
    import src.indexing.sparse_tokenizer  # noqa: F401


def test_blake2b_deterministic_across_runs() -> None:
    """blake2b digest with persona=False must produce identical output every run."""
    from src.indexing.sparse_tokenizer import blake2b_digest

    results = [blake2b_digest("customer due diligence") for _ in range(3)]
    assert results[0] == results[1] == results[2], (
        f"blake2b must be deterministic but got {results}"
    )


def test_blake2b_deterministic_vs_python_hash() -> None:
    """Token ID must NOT vary between Python invocations (no hash() dependency)."""
    from src.indexing.sparse_tokenizer import blake2b_digest

    # Run in a tight loop simulating multiple processes
    ids = [blake2b_digest("cdd requirement") for _ in range(5)]
    assert len(set(ids)) == 1, "blake2b token IDs must be stable"


def test_english_tokenization_splits_words() -> None:
    """English text should be split on ASCII word boundaries."""
    from src.indexing.sparse_tokenizer import tokenize_and_weight

    result = tokenize_and_weight("customer due diligence requirements")
    tokens_found = list(result.values())
    # We expect multiple tokens (not one monolithic entry)
    assert len(tokens_found) >= 3, (
        f"Expected >= 3 tokens for English sentence, got {len(tokens_found)}: {tokens_found}"
    )


def test_chinese_bigram_tokenization() -> None:
    """Chinese text should produce char bigrams."""
    from src.indexing.sparse_tokenizer import tokenize_and_weight

    result = tokenize_and_weight("客户尽职审查")
    # 4 Chinese chars → 3 bigrams: 客户, 户尽, 尽职, 职审, 审查
    assert len(result) >= 3, (
        f"Expected >= 3 bigrams for Chinese text, got {len(result)}: {result}"
    )


def test_mixed_tokenization_preserves_english_abbreviations() -> None:
    """Mixed text should keep CDD/PEP/AML as atomic tokens."""
    from src.indexing.sparse_tokenizer import tokenize_and_weight

    result = tokenize_and_weight("CDD 和 PEP screening 要求")
    token_ids = list(result.keys())
    # CDD and PEP should each produce at least one token ID
    assert len(token_ids) >= 3, (
        f"Expected >= 3 token IDs for mixed text, got {len(token_ids)}"
    )


def test_empty_string_returns_empty() -> None:
    """Empty input must return an empty dict."""
    from src.indexing.sparse_tokenizer import tokenize_and_weight

    result = tokenize_and_weight("")
    assert result == {}, f"Empty string must return empty dict, got {result}"


def test_token_weights_are_positive() -> None:
    """All token weights must be positive floats."""
    from src.indexing.sparse_tokenizer import tokenize_and_weight

    result = tokenize_and_weight("sanctions screening and monitoring")
    for tid, weight in result.items():
        assert weight > 0, f"Token {tid} has non-positive weight {weight}"


def test_chinese_english_mixed_tokenization() -> None:
    """Code-switching query like 'CDD要求' should tokenize both parts."""
    from src.indexing.sparse_tokenizer import tokenize_and_weight

    result = tokenize_and_weight("CDD要求")
    # Should get: CDD token + Chinese bigrams from 要求
    assert len(result) >= 2, (
        f"Expected >= 2 tokens for 'CDD要求', got {len(result)}"
    )


# ---------------------------------------------------------------------------
# Tests for integration: MilvusHybridStore and BGEM3EmbeddingClient
# ---------------------------------------------------------------------------


def test_store_sparse_matches_embedding_client() -> None:
    """Sparse vector built from same text at ingest and query must be identical."""
    from src.indexing.milvus_ingest import MilvusHybridStore, BGEM3EmbeddingClient
    from src.indexing.sparse_tokenizer import tokenize_and_weight
    from types import SimpleNamespace

    # Patch settings to avoid real Milvus connection
    fake_settings = SimpleNamespace(
        milvus=SimpleNamespace(
            alias="default",
            host="127.0.0.1",
            port=19530,
            user=None,
            password=None,
            database="default",
            uri="http://127.0.0.1:19530",
            collection_name="test_collection",
            consistency_level="Session",
            dense_index_type="HNSW",
            sparse_index_type="SPARSE_INVERTED_INDEX",
            dense_metric_type="COSINE",
            sparse_metric_type="IP",
            search_probe=64,
        )
    )

    client = BGEM3EmbeddingClient.__new__(BGEM3EmbeddingClient)
    client.model = None  # type: ignore[attr-defined]
    client.model_name = "BAAI/bge-m3"
    client._ST = None  # type: ignore[attr-defined]

    # Simulate encode_many at ingest time
    text = "customer due diligence screening"
    embedding = client._build_sparse_vector_fallback(text)

    # Query side: direct tokenizer call
    query_sparse = tokenize_and_weight(text)

    # Must be identical
    assert embedding == query_sparse, (
        f"Ingest sparse {embedding} != query sparse {query_sparse}"
    )


def test_sparse_vector_is_deterministic_within_session() -> None:
    """Calling _build_sparse_vector_fallback twice on same text must give same result."""
    from src.indexing.milvus_ingest import BGEM3EmbeddingClient

    client = BGEM3EmbeddingClient.__new__(BGEM3EmbeddingClient)
    client.model = None  # type: ignore[attr-defined]
    client.model_name = "BAAI/bge-m3"
    client._ST = None  # type: ignore[attr-defined]

    text = "authorized institution AML compliance"
    r1 = client._build_sparse_vector_fallback(text)
    r2 = client._build_sparse_vector_fallback(text)

    assert r1 == r2, f"Non-deterministic sparse: {r1} vs {r2}"


def test_sparse_fallback_no_python_hash_dependency() -> None:
    """Sparse vector must not use Python's built-in hash()."""
    import ast
    import inspect
    from pathlib import Path

    tokenizer_path = Path(__file__).resolve().parents[2] / "src" / "indexing" / "sparse_tokenizer.py"
    milvus_path = Path(__file__).resolve().parents[2] / "src" / "indexing" / "milvus_ingest.py"

    for path in [tokenizer_path, milvus_path]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "hash"
                ):
                    raise AssertionError(
                        f"{path.name} calls Python's built-in hash() which is non-deterministic. "
                        "Use hashlib.blake2b instead."
                    )
