from __future__ import annotations

from types import SimpleNamespace

from src.evaluation.annotate_gt import _JUDGE_TEXT_LIMIT, build_annotation_prompt, search_chunks


class _EmbeddingClient:
    def encode(self, query: str, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(dense_vector=[0.1], sparse_vector={1: 0.2})


class _Store:
    def hybrid_search(self, **kwargs):
        self.top_k = kwargs["top_k"]
        dense = [
            {"chunk_id": "dense-only", "text": "d" * 1500},
            {"chunk_id": "shared", "text": "shared"},
        ]
        sparse = [
            {"chunk_id": "sparse-only", "text": "sparse"},
            {"chunk_id": "shared", "text": "shared"},
        ]
        return dense, sparse


def test_search_chunks_rrf_uses_dense_and_sparse_results() -> None:
    store = _Store()

    chunks = search_chunks("query", _EmbeddingClient(), store, top_k=3)

    assert store.top_k == 20
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "shared",
        "dense-only",
        "sparse-only",
    ]
    assert len(chunks[1]["text"]) == _JUDGE_TEXT_LIMIT


def test_annotation_prompt_includes_reference_answer_and_selection_cap() -> None:
    prompt = build_annotation_prompt(
        "When is CDD required?",
        [{"chunk_id": "child-1", "source_file": "guide.pdf", "page_number": 20, "text": "Evidence"}],
        "CDD is required before establishing a business relationship.",
    )

    assert "Human Reference Answer" in prompt
    assert "CDD is required before establishing" in prompt
    assert "no more than 3 chunks" in prompt
