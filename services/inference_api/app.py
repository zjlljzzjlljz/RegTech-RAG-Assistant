from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


SERVICE_MODE = os.getenv("SERVICE_MODE", "embedding").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
DEVICE = int(os.getenv("DEVICE", "0"))

app = FastAPI(title=f"RegTech {SERVICE_MODE} inference")

_READINESS_NOT_STARTED = "not_started"
_READINESS_LOADING = "loading"
_READINESS_READY = "ready"
_READINESS_FAILED = "failed"
_readiness_state = _READINESS_NOT_STARTED
_readiness_lock = threading.Lock()
_model: Any | None = None
_readiness_generation = 0


def _load_model_in_background(generation: int) -> None:
    global _model, _readiness_state
    try:
        model = load_model()
    except Exception:
        with _readiness_lock:
            if generation == _readiness_generation:
                _model = None
                _readiness_state = _READINESS_FAILED
    else:
        with _readiness_lock:
            if generation == _readiness_generation:
                _model = model
                _readiness_state = _READINESS_READY


def _model_load_failed() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "status": "failed",
            "code": "MODEL_LOAD_FAILED",
            "message": "Model initialization failed",
        },
    )


def _get_model() -> Any:
    global _readiness_state
    with _readiness_lock:
        if _readiness_state == _READINESS_NOT_STARTED:
            _readiness_state = _READINESS_LOADING
            generation = _readiness_generation
            threading.Thread(
                target=_load_model_in_background, args=(generation,), daemon=True
            ).start()
        state = _readiness_state
        model = _model

    if state == _READINESS_READY:
        return model
    if state == _READINESS_FAILED:
        raise _model_load_failed()
    raise HTTPException(
        status_code=503,
        detail={"status": "loading", "code": "MODEL_NOT_READY"},
    )


def _reset_readiness_state() -> None:
    global _model, _readiness_generation, _readiness_state
    with _readiness_lock:
        _readiness_generation += 1
        _readiness_state = _READINESS_NOT_STARTED
        _model = None


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
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    _get_model()
    return {"status": "ready", "mode": SERVICE_MODE, "model": "[redacted]"}


@app.post("/encode")
def encode(request: EncodeRequest) -> dict[str, Any]:
    if SERVICE_MODE != "embedding":
        raise HTTPException(status_code=404, detail="Embedding endpoint disabled")
    result = _get_model().encode(
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
    raw_scores = _get_model().compute_score(pairs, normalize=True)
    scores = [float(raw_scores)] if np.isscalar(raw_scores) else [float(value) for value in raw_scores]
    return {"model": MODEL_NAME, "scores": scores}


@app.post("/nli")
def nli(request: NLIRequest) -> dict[str, Any]:
    if SERVICE_MODE != "nli":
        raise HTTPException(status_code=404, detail="NLI endpoint disabled")
    result = _get_model()({"text": request.premise, "text_pair": request.hypothesis})
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
