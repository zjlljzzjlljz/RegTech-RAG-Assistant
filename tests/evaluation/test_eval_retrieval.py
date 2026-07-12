from pathlib import Path
from types import SimpleNamespace
import sys
import types

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.evaluation.eval_retrieval as eval_retrieval


def test_parse_args_defaults_to_builtin_test_queries_file() -> None:
    args = eval_retrieval.parse_args([])

    assert Path(args.queries_file) == Path(eval_retrieval.__file__).resolve().parent / "test_queries_v2.json"


def test_parse_args_fusion_default_is_rrf() -> None:
    args = eval_retrieval.parse_args([])
    assert args.fusion == "rrf"


def test_parse_args_fusion_choices() -> None:
    for mode in ["all", "weighted-sweep", "rrf", "dedup", "sparse-only", "dense-only"]:
        args = eval_retrieval.parse_args(["--fusion", mode])
        assert args.fusion == mode


def test_parse_args_with_rerank_flag() -> None:
    args = eval_retrieval.parse_args(["--with-rerank"])
    assert args.with_rerank is True


def test_parse_args_no_rerank_flag() -> None:
    args = eval_retrieval.parse_args(["--no-rerank"])
    assert args.no_rerank is True


def test_main_uses_custom_queries_file(monkeypatch, tmp_path) -> None:
    custom_queries = tmp_path / "custom_queries.json"
    custom_queries.write_text("[]", encoding="utf-8")

    loaded_paths: list[Path] = []
    evaluated_queries: list[list[dict]] = []
    printed_results: list[list[dict]] = []

    fake_settings = SimpleNamespace(
        app=SimpleNamespace(log_level="INFO"),
        inference=SimpleNamespace(embedding_model_name="fake-embedding-model"),
        retrieval=SimpleNamespace(
            rrf_k=60,
            rrf_top_k=20,
            rerank_top_k=8,
            rerank_score_threshold=0.25,
            dense_top_k=50,
            sparse_top_k=50,
        ),
    )

    monkeypatch.setattr(eval_retrieval, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(eval_retrieval, "configure_logging", lambda level: None)
    monkeypatch.setattr(eval_retrieval, "load_test_queries", lambda path: loaded_paths.append(path) or [{"query": "q", "ground_truth_chunk_ids": []}])
    monkeypatch.setattr(eval_retrieval, "MilvusHybridStore", lambda settings: {"settings": settings})
    monkeypatch.setattr(eval_retrieval, "BGEM3EmbeddingClient", lambda model_name: {"model_name": model_name})
    monkeypatch.setattr(
        eval_retrieval,
        "evaluate",
        lambda test_queries, embedding_client, store, settings, fusion="rrf", with_rerank=False: (
            evaluated_queries.append(test_queries) or [{"query": "q", "ground_truth_chunk_ids": [], "retrieved_chunk_ids": []}]
        ),
    )
    monkeypatch.setattr(eval_retrieval, "print_report", lambda results, fusion="rrf", with_rerank=False: printed_results.append(results))

    eval_retrieval.main(["--queries-file", str(custom_queries)])

    assert loaded_paths == [custom_queries]
    assert evaluated_queries == [[{"query": "q", "ground_truth_chunk_ids": []}]]
    assert printed_results == [[{"query": "q", "ground_truth_chunk_ids": [], "retrieved_chunk_ids": []}]]


def test_search_single_query_uses_supplied_reranker() -> None:
    class FakeEmbeddingClient:
        def encode(self, query: str, prompt: str):
            assert query == "find me"
            assert prompt.startswith("Represent this sentence")
            return SimpleNamespace(dense_vector=[0.1], sparse_vector={"token": 1.0})

    class FakeStore:
        def dense_only_search(self, dense_vector, top_k, output_fields, filters):
            assert dense_vector == [0.1]
            assert top_k == 8
            return [
                {
                    "chunk_id": "chunk-1",
                    "parent_id": "parent-1",
                    "text": "retrieved text",
                    "source_file": "doc.pdf",
                    "page_number": 12,
                    "chunk_type": "child",
                    "score": 0.9,
                    "metadata": {"section": "A"},
                }
            ]

    class FakeReranker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, float, list[str]]] = []

        def rerank(self, query, chunks, top_k, threshold):
            self.calls.append((query, top_k, threshold, [chunk.chunk_id for chunk in chunks]))
            return chunks

    settings = SimpleNamespace(
        retrieval=SimpleNamespace(
            rrf_k=60,
            rrf_top_k=20,
            rerank_top_k=8,
            rerank_score_threshold=0.25,
            dense_top_k=50,
            sparse_top_k=50,
        )
    )
    reranker = FakeReranker()

    chunk_ids = eval_retrieval.search_single_query(
        "find me",
        FakeEmbeddingClient(),
        FakeStore(),
        settings,
        fusion="dense-only",
        with_rerank=True,
        reranker=reranker,
    )

    assert chunk_ids == ["chunk-1"]
    assert reranker.calls == [("find me", 8, 0.25, ["chunk-1"])]


def test_evaluate_constructs_reranker_once_and_reuses_it(monkeypatch) -> None:
    constructed_models: list[str] = []
    seen_rerankers: list[object | None] = []

    class FakeCrossEncoderRerankerClient:
        def __init__(self, model_name: str) -> None:
            constructed_models.append(model_name)

    fake_module = types.ModuleType("src.retrieval.query_pipeline")
    fake_module.CrossEncoderRerankerClient = FakeCrossEncoderRerankerClient
    monkeypatch.setitem(sys.modules, "src.retrieval.query_pipeline", fake_module)

    def fake_search_single_query(
        query,
        embedding_client,
        store,
        settings,
        fusion="rrf",
        with_rerank=False,
        reranker=None,
    ):
        assert with_rerank is True
        seen_rerankers.append(reranker)
        return [f"retrieved-{query}"]

    monkeypatch.setattr(eval_retrieval, "search_single_query", fake_search_single_query)

    settings = SimpleNamespace(
        inference=SimpleNamespace(reranker_model_name="fake-reranker"),
        retrieval=SimpleNamespace(),
    )

    results = eval_retrieval.evaluate(
        [
            {"query": "q1", "ground_truth_chunk_ids": []},
            {"query": "q2", "ground_truth_chunk_ids": []},
        ],
        embedding_client=object(),
        store=object(),
        settings=settings,
        fusion="rrf",
        with_rerank=True,
    )

    assert [result["retrieved_chunk_ids"] for result in results] == [["retrieved-q1"], ["retrieved-q2"]]
    assert constructed_models == ["fake-reranker"]
    assert len(seen_rerankers) == 2
    assert seen_rerankers[0] is not None
    assert seen_rerankers[0] is seen_rerankers[1]


def test_evaluate_ablation_encodes_once_and_builds_all_modes() -> None:
    class FakeEmbeddingClient:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, query, prompt):
            self.calls += 1
            return SimpleNamespace(dense_vector=[0.1], sparse_vector={1: 0.2})

    class FakeStore:
        def hybrid_search(self, **kwargs):
            return (
                [{"chunk_id": "dense"}, {"chunk_id": "shared"}],
                [{"chunk_id": "sparse"}, {"chunk_id": "shared"}],
            )

    settings = SimpleNamespace(
        retrieval=SimpleNamespace(dense_top_k=2, sparse_top_k=2, rrf_top_k=2, rrf_k=60)
    )
    embedding_client = FakeEmbeddingClient()

    results = eval_retrieval.evaluate_ablation(
        [{"query": "q", "ground_truth_chunk_ids": ["shared"]}],
        embedding_client,
        FakeStore(),
        settings,
    )

    assert embedding_client.calls == 1
    assert set(results) == {"dense_only", "sparse_only", "dedup", "rrf"}
    assert results["dense_only"][0]["retrieved_chunk_ids"] == ["dense", "shared"]
    assert results["sparse_only"][0]["retrieved_chunk_ids"] == ["sparse", "shared"]
    assert results["rrf"][0]["retrieved_chunk_ids"][0] == "shared"


def test_weighted_rrf_prefers_dense_when_dense_weight_is_higher() -> None:
    dense = [{"chunk_id": "other-dense"}, {"chunk_id": "dense"}]
    sparse = [{"chunk_id": "sparse"}]

    equal = eval_retrieval._weighted_rank_fusion(dense, sparse, 60, 3, 1.0, 1.0)
    dense_weighted = eval_retrieval._weighted_rank_fusion(dense, sparse, 60, 3, 4.0, 1.0)

    assert equal[0]["chunk_id"] == "other-dense"
    assert equal[1]["chunk_id"] == "sparse"
    assert [hit["chunk_id"] for hit in dense_weighted].index("dense") < [
        hit["chunk_id"] for hit in dense_weighted
    ].index("sparse")
