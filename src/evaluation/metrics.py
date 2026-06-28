"""Retrieval evaluation metrics for compliance RAG pipeline."""

from __future__ import annotations


def calculate_hit_at_k(results: list[dict], k: int) -> float:
    """Calculate Hit@K — fraction of queries with at least one relevant chunk in top-K results.

    Parameters
    ----------
    results : list[dict]
        Each dict must have:
        - ground_truth_chunk_ids: list[str] — relevant chunk IDs
        - retrieved_chunk_ids: list[str] — retrieved chunk IDs in rank order
    k : int
        Number of top results to consider.

    Returns
    -------
    float
        Hit@K score in [0.0, 1.0].
    """
    if not results:
        return 0.0

    hits = 0
    for result in results:
        ground_truth = set(result.get("ground_truth_chunk_ids", []))
        if not ground_truth:
            continue
        retrieved_top_k = result.get("retrieved_chunk_ids", [])[:k]
        if ground_truth & set(retrieved_top_k):
            hits += 1

    total = len([r for r in results if r.get("ground_truth_chunk_ids")])
    if total == 0:
        return 0.0
    return hits / total


def calculate_mrr(results: list[dict]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    For each query, MRR = 1 / rank_of_first_relevant_chunk.
    If no relevant chunk found, contribution is 0.

    Parameters
    ----------
    results : list[dict]
        Each dict must have:
        - ground_truth_chunk_ids: list[str]
        - retrieved_chunk_ids: list[str]

    Returns
    -------
    float
        MRR score in [0.0, 1.0].
    """
    if not results:
        return 0.0

    reciprocal_ranks: list[float] = []
    for result in results:
        ground_truth = set(result.get("ground_truth_chunk_ids", []))
        retrieved = result.get("retrieved_chunk_ids", [])
        if not ground_truth:
            continue
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in ground_truth:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


__all__ = [
    "calculate_hit_at_k",
    "calculate_mrr",
]
