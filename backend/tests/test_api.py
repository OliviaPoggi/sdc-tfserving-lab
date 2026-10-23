"""Backend tests.

TensorFlow Serving is mocked with respx, so the suite runs without the model
server, without TensorFlow and without network access.
"""
import io
import json

import httpx
import numpy as np
import pytest
import respx
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings, get_settings
from app.main import app
from app.services.preprocessing import InvalidImageError, preprocess

TF_SERVING_URL = "http://tf-serving-test:8501"
MODEL_NAME = "mobilenet"
PREDICT_URL = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}:predict"
STATUS_URL = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}"


def make_image_bytes(size=(400, 300), colour=(120, 80, 200), fmt="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format=fmt)
    return buffer.getvalue()


def fake_scores(winner_index: int = 207) -> list[float]:
    scores = np.full(1000, 0.0001, dtype=float)
    scores[winner_index] = 0.92
    return scores.tolist()


@pytest.fixture
def labels_file(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"207": "golden_retriever", "208": "Labrador_retriever"}))
    return str(path)


@pytest.fixture
def client(labels_file):
    settings = Settings(
        tf_serving_url=TF_SERVING_URL,
        model_name=MODEL_NAME,
        labels_path=labels_file,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ----------------------------------------------------------- preprocessing


def test_preprocess_shape_and_range():
    instance = preprocess(make_image_bytes(), image_size=224)
    array = np.array(instance)
    assert array.shape == (224, 224, 3)
    assert array.min() >= -1.0 and array.max() <= 1.0


def test_preprocess_converts_greyscale_to_rgb():
    buffer = io.BytesIO()
    Image.new("L", (50, 50), 128).save(buffer, format="PNG")
    array = np.array(preprocess(buffer.getvalue(), image_size=224))
    assert array.shape == (224, 224, 3)


def test_preprocess_rejects_non_image():
    with pytest.raises(InvalidImageError):
        preprocess(b"this is definitely not an image")


# ------------------------------------------------------------------ health


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ----------------------------------------------------------------- predict


@respx.mock
def test_predict_returns_top5_with_labels(client):
    respx.post(PREDICT_URL).mock(
        return_value=httpx.Response(200, json={"predictions": [fake_scores(207)]})
    )

    response = client.post(
        "/predict",
        files={"file": ("dog.png", make_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "dog.png"
    assert payload["model_name"] == MODEL_NAME
    assert len(payload["predictions"]) == 5

    best = payload["predictions"][0]
    assert best["class_index"] == 207
    assert best["label"] == "golden_retriever"
    assert best["probability"] == pytest.approx(0.92)

    probabilities = [p["probability"] for p in payload["predictions"]]
    assert probabilities == sorted(probabilities, reverse=True)


@respx.mock
def test_predict_sends_correctly_shaped_instance(client):
    route = respx.post(PREDICT_URL).mock(
        return_value=httpx.Response(200, json={"predictions": [fake_scores()]})
    )

    client.post("/predict", files={"file": ("x.png", make_image_bytes(), "image/png")})

    sent = json.loads(route.calls[0].request.content)
    assert "instances" in sent
    assert np.array(sent["instances"]).shape == (1, 224, 224, 3)


@respx.mock
def test_predict_unknown_class_falls_back_to_generic_label(client):
    # Index 999 is not in the labels fixture.
    respx.post(PREDICT_URL).mock(
        return_value=httpx.Response(200, json={"predictions": [fake_scores(999)]})
    )

    response = client.post("/predict", files={"file": ("x.png", make_image_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["predictions"][0]["label"] == "class_999"


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 422


def test_predict_rejects_empty_file(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 422


@respx.mock
def test_predict_returns_503_when_model_server_is_down(client):
    respx.post(PREDICT_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    response = client.post("/predict", files={"file": ("x.png", make_image_bytes(), "image/png")})
    assert response.status_code == 503
    assert "Could not reach TensorFlow Serving" in response.json()["detail"]


@respx.mock
def test_predict_surfaces_model_server_error(client):
    respx.post(PREDICT_URL).mock(
        return_value=httpx.Response(400, text="model not found")
    )

    response = client.post("/predict", files={"file": ("x.png", make_image_bytes(), "image/png")})
    assert response.status_code == 503


# ------------------------------------------------------------ model status


@respx.mock
def test_model_status_available(client):
    respx.get(STATUS_URL).mock(
        return_value=httpx.Response(
            200, json={"model_version_status": [{"version": "1", "state": "AVAILABLE"}]}
        )
    )

    response = client.get("/model/status")
    assert response.status_code == 200
    assert response.json()["available"] is True


@respx.mock
def test_model_status_unreachable(client):
    respx.get(STATUS_URL).mock(side_effect=httpx.ConnectError("refused"))

    response = client.get("/model/status")
    assert response.status_code == 200
    assert response.json()["available"] is False
