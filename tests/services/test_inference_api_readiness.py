import sys
import threading
import time
import types

import pytest

try:
    import fastapi  # noqa: F401
except ModuleNotFoundError:
    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, **_kwargs):
            pass

        def get(self, _path):
            return lambda function: function

        def post(self, _path):
            return lambda function: function

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi

from services.inference_api import app as inference_app


def test_health_is_liveness_only(monkeypatch):
    def fail_if_loaded():
        raise AssertionError("health must not load the model")

    monkeypatch.setattr(inference_app, "load_model", fail_if_loaded)
    inference_app._reset_readiness_state()

    assert inference_app.health() == {"status": "ok"}
    assert inference_app._readiness_state == inference_app._READINESS_NOT_STARTED


def test_first_and_repeated_ready_start_one_background_load(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def fake_loader():
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return object()

    monkeypatch.setattr(inference_app, "load_model", fake_loader)
    inference_app._reset_readiness_state()

    with pytest.raises(inference_app.HTTPException) as first:
        inference_app.ready()
    assert first.value.status_code == 503
    assert first.value.detail == {"status": "loading", "code": "MODEL_NOT_READY"}
    assert started.wait(timeout=1)

    with pytest.raises(inference_app.HTTPException) as repeated:
        inference_app.ready()
    assert repeated.value.status_code == 503
    assert repeated.value.detail == {"status": "loading", "code": "MODEL_NOT_READY"}
    assert calls == 1

    release.set()


def test_successful_load_becomes_ready_without_reloading(monkeypatch):
    completed = threading.Event()
    calls = 0

    def fake_loader():
        nonlocal calls
        calls += 1
        completed.set()
        return object()

    monkeypatch.setattr(inference_app, "load_model", fake_loader)
    inference_app._reset_readiness_state()

    with pytest.raises(inference_app.HTTPException):
        inference_app.ready()
    assert completed.wait(timeout=1)

    deadline = time.monotonic() + 1
    while True:
        try:
            response = inference_app.ready()
            break
        except inference_app.HTTPException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)
    assert response == {
        "status": "ready",
        "mode": inference_app.SERVICE_MODE,
        "model": "[redacted]",
    }
    assert inference_app.ready() == response
    assert calls == 1


def test_failed_load_is_sanitized_and_cached(monkeypatch):
    completed = threading.Event()
    calls = 0

    def fake_loader():
        nonlocal calls
        calls += 1
        try:
            raise RuntimeError("secret token and internal model path")
        finally:
            completed.set()

    monkeypatch.setattr(inference_app, "load_model", fake_loader)
    inference_app._reset_readiness_state()

    with pytest.raises(inference_app.HTTPException):
        inference_app.ready()
    assert completed.wait(timeout=1)

    deadline = time.monotonic() + 1
    while True:
        try:
            inference_app.ready()
        except inference_app.HTTPException as error:
            if error.detail.get("status") == "failed":
                failure = error
                break
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)

    assert failure.status_code == 503
    assert failure.detail == {
        "status": "failed",
        "code": "MODEL_LOAD_FAILED",
        "message": "Model initialization failed",
    }
    assert "secret" not in str(failure.detail)

    with pytest.raises(inference_app.HTTPException) as repeated:
        inference_app.ready()
    assert repeated.value.detail == failure.detail

    monkeypatch.setattr(inference_app, "SERVICE_MODE", "embedding")
    with pytest.raises(inference_app.HTTPException) as business:
        inference_app.encode(inference_app.EncodeRequest(texts=["x"]))
    assert business.value.detail == failure.detail
    assert calls == 1


def test_ready_loading_then_business_does_not_start_second_load(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def fake_loader():
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return object()

    monkeypatch.setattr(inference_app, "load_model", fake_loader)
    monkeypatch.setattr(inference_app, "SERVICE_MODE", "embedding")
    inference_app._reset_readiness_state()

    with pytest.raises(inference_app.HTTPException):
        inference_app.ready()
    assert started.wait(timeout=1)

    with pytest.raises(inference_app.HTTPException) as error:
        inference_app.encode(inference_app.EncodeRequest(texts=["x"]))
    assert error.value.status_code == 503
    assert error.value.detail == {"status": "loading", "code": "MODEL_NOT_READY"}
    assert calls == 1
    release.set()


def test_concurrent_business_requests_start_one_background_load(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def fake_loader():
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return object()

    monkeypatch.setattr(inference_app, "load_model", fake_loader)
    monkeypatch.setattr(inference_app, "SERVICE_MODE", "embedding")
    inference_app._reset_readiness_state()
    barrier = threading.Barrier(9)
    errors = []

    def invoke():
        barrier.wait()
        try:
            inference_app.encode(inference_app.EncodeRequest(texts=["x"]))
        except inference_app.HTTPException as error:
            errors.append(error)

    threads = [threading.Thread(target=invoke) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1)

    assert started.wait(timeout=1)
    assert calls == 1
    assert len(errors) == 8
    assert all(error.detail == {"status": "loading", "code": "MODEL_NOT_READY"} for error in errors)
    release.set()


def test_loaded_model_instance_is_shared_by_all_inference_endpoints(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.operations = []

        def encode(self, texts, **_kwargs):
            self.operations.append(("encode", texts))
            return {"dense_vecs": [[1.0]], "lexical_weights": [{1: 0.5}]}

        def compute_score(self, pairs, normalize):
            self.operations.append(("rerank", pairs, normalize))
            return [0.25]

        def __call__(self, item):
            self.operations.append(("nli", item))
            return [[{"label": "entailment", "score": 0.75}]]

    model = FakeModel()
    completed = threading.Event()

    def fake_loader():
        completed.set()
        return model

    monkeypatch.setattr(inference_app, "load_model", fake_loader)
    inference_app._reset_readiness_state()
    with pytest.raises(inference_app.HTTPException):
        inference_app.ready()
    assert completed.wait(timeout=1)
    deadline = time.monotonic() + 1
    while inference_app._readiness_state != inference_app._READINESS_READY:
        assert time.monotonic() < deadline
        time.sleep(0.001)

    monkeypatch.setattr(inference_app, "SERVICE_MODE", "embedding")
    assert inference_app.encode(inference_app.EncodeRequest(texts=["x"]))["embeddings"]
    monkeypatch.setattr(inference_app, "SERVICE_MODE", "reranker")
    assert inference_app.rerank(inference_app.RerankRequest(query="q", documents=["d"]))["scores"] == [0.25]
    monkeypatch.setattr(inference_app, "SERVICE_MODE", "nli")
    assert inference_app.nli(inference_app.NLIRequest(premise="p", hypothesis="h"))["entailment"] == 0.75

    assert inference_app._model is model
    assert [operation[0] for operation in model.operations] == ["encode", "rerank", "nli"]


def test_reset_clears_ready_model_reference(monkeypatch):
    model = object()
    completed = threading.Event()

    def fake_loader():
        completed.set()
        return model

    monkeypatch.setattr(inference_app, "load_model", fake_loader)
    inference_app._reset_readiness_state()
    with pytest.raises(inference_app.HTTPException):
        inference_app.ready()
    assert completed.wait(timeout=1)
    deadline = time.monotonic() + 1
    while inference_app._readiness_state != inference_app._READINESS_READY:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert inference_app._model is model

    inference_app._reset_readiness_state()

    assert inference_app._readiness_state == inference_app._READINESS_NOT_STARTED
    assert inference_app._model is None


def test_concurrent_ready_requests_start_one_background_load(monkeypatch):
    release = threading.Event()
    calls = 0

    def fake_loader():
        nonlocal calls
        calls += 1
        assert release.wait(timeout=2)
        return object()

    monkeypatch.setattr(inference_app, "load_model", fake_loader)
    inference_app._reset_readiness_state()
    barrier = threading.Barrier(9)
    errors = []

    def invoke():
        barrier.wait()
        try:
            inference_app.ready()
        except inference_app.HTTPException as error:
            errors.append(error)

    threads = [threading.Thread(target=invoke) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1)

    assert calls == 1
    assert len(errors) == 8
    assert all(error.detail == {"status": "loading", "code": "MODEL_NOT_READY"} for error in errors)
    release.set()
