import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = PROJECT_DIR / "chroma_db"
HYBRID_SCRIPT_PATH = PROJECT_DIR / "5_hybrid_retrieval.py"
OUTPUT_JSON_PATH = PROJECT_DIR / "comparison_results.json"

QUERIES = [
    "What are the customer due diligence requirements?",
    "Section 4.1 AML/CFT obligations",
    "risk assessment procedures for money laundering",
]


def load_hybrid_retrieve():
    spec = importlib.util.spec_from_file_location("hybrid_retrieval_module", HYBRID_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load hybrid retrieval module from: {HYBRID_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hybrid_retrieve = getattr(module, "hybrid_retrieve", None)
    if hybrid_retrieve is None:
        raise AttributeError("5_hybrid_retrieval.py does not define hybrid_retrieve")

    return hybrid_retrieve


def build_old_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name="regtech_pdf_docs",
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )


def normalize_old_results(old_docs: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
    normalized_results: List[Dict[str, Any]] = []

    for index, (doc, score) in enumerate(old_docs, start=1):
        raw_page = doc.metadata.get("page")
        if isinstance(raw_page, bool):
            page_value = int(raw_page)
        elif isinstance(raw_page, (int, float)):
            page_value = int(raw_page)
        else:
            page_value = raw_page

        if isinstance(page_value, int):
            page_display = str(page_value + 1)
        else:
            page_display = str(page_value) if page_value not in (None, "") else "N/A"

        normalized_results.append(
            {
                "label": f"E{index}",
                "source_file": doc.metadata.get("source_file", "unknown"),
                "page": page_value,
                "page_display": page_display,
                "score": float(score),
                "text": doc.page_content,
            }
        )

    return normalized_results


def preview(text: str, limit: int = 120) -> str:
    normalized_text = (text or "").replace("\n", " ").strip()
    if len(normalized_text) <= limit:
        return normalized_text
    return f"{normalized_text[:limit]}..."


def main() -> int:
    hybrid_retrieve = load_hybrid_retrieve()
    old_vector_store = build_old_vector_store()

    all_results: List[Dict[str, Any]] = []

    for query in QUERIES:
        print(f"\n{'=' * 80}")
        print(f"QUERY: {query}")
        print(f"{'=' * 80}")

        start_time = time.perf_counter()
        new_results = hybrid_retrieve(query, DB_DIR, top_k=3)
        hybrid_elapsed = time.perf_counter() - start_time

        start_time = time.perf_counter()
        old_docs = old_vector_store.similarity_search_with_score(query, k=3)
        old_elapsed = time.perf_counter() - start_time
        old_results = normalize_old_results(old_docs)

        print(f"\n--- NEW (Hybrid Parent-Child) --- {hybrid_elapsed:.2f}s")
        for result in new_results:
            print(
                f"  {result['label']} | {result['source_file']} p.{result['page_display']} | "
                f"score={result['score']} | {preview(result['text'])}"
            )

        print(f"\n--- OLD (Flat Dense Only) --- {old_elapsed:.2f}s")
        for result in old_results:
            print(
                f"  {result['label']} | {result['source_file']} p.{result['page_display']} | "
                f"score={result['score']:.4f} | {preview(result['text'])}"
            )

        all_results.append(
            {
                "query": query,
                "hybrid_elapsed": hybrid_elapsed,
                "old_elapsed": old_elapsed,
                "hybrid_results": new_results,
                "old_results": old_results,
            }
        )

    OUTPUT_JSON_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved structured results to: {OUTPUT_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
