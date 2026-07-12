from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


SERVICE_MODE = os.getenv("SERVICE_MODE", "embedding").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
DEVICE = int(os.getenv("DEVICE", "0"))

app = FastAPI(title=f"RegTech {SERVICE_MODE} inference")


class EncodeRequest(BaseModel):
    texts: list[str]
    batch_size: int = 16
    max_length: int = 2048


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


class NLIRequest(BaseModel):
    premise: str
    hypothesis: str


@lru_cache(maxsize=1)
def load_model() -> Any:
    if SERVICE_MODE == "embedding":
        from FlagEmbedding import BGEM3FlagModel

        return BGEM3FlagModel(MODEL_NAME, use_fp16=USE_FP16)
    if SERVICE_MODE == "reranker":
        from FlagEmbedding import FlagReranker

        return FlagReranker(MODEL_NAME, use_fp16=USE_FP16)
    if SERVICE_MODE == "nli":
        from transformers import pipeline

        return pipeline("text-classification", model=MODEL_NAME, top_k=None, device=DEVICE)
    raise RuntimeError(f"Unsupported SERVICE_MODE: {SERVICE_MODE}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": SERVICE_MODE, "model": MODEL_NAME}


@app.post("/encode")
def encode(request: EncodeRequest) -> dict[str, Any]:
    if SERVICE_MODE != "embedding":
        raise HTTPException(status_code=404, detail="Embedding endpoint disabled")
    result = load_model().encode(
        request.texts,
        batch_size=request.batch_size,
        max_length=request.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    embeddings = []
    for dense, sparse in zip(result["dense_vecs"], result["lexical_weights"]):
        embeddings.append(
            {
                "dense_vector": np.asarray(dense, dtype=np.float32).tolist(),
                "sparse_vector": {str(key): float(value) for key, value in sparse.items()},
            }
        )
    return {"model": MODEL_NAME, "embeddings": embeddings}


@app.post("/rerank")
def rerank(request: RerankRequest) -> dict[str, Any]:
    if SERVICE_MODE != "reranker":
        raise HTTPException(status_code=404, detail="Rerank endpoint disabled")
    pairs = [[request.query, document] for document in request.documents]
    raw_scores = load_model().compute_score(pairs, normalize=True)
    scores = [float(raw_scores)] if np.isscalar(raw_scores) else [float(value) for value in raw_scores]
    return {"model": MODEL_NAME, "scores": scores}


@app.post("/nli")
def nli(request: NLIRequest) -> dict[str, Any]:
    if SERVICE_MODE != "nli":
        raise HTTPException(status_code=404, detail="NLI endpoint disabled")
    result = load_model()({"text": request.premise, "text_pair": request.hypothesis})
    labels = result[0] if result and isinstance(result[0], list) else result
    entailment = 0.0
    contradiction = 0.0
    for item in labels:
        label = str(item.get("label", "")).lower()
        if "entail" in label or label in {"label_2", "2"}:
            entailment = float(item.get("score", 0.0))
        if "contrad" in label or label in {"label_0", "0"}:
            contradiction = float(item.get("score", 0.0))
    return {"model": MODEL_NAME, "entailment": entailment, "contradiction": contradiction}
