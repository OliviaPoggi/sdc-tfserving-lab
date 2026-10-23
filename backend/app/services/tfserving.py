"""Client for the TensorFlow Serving REST API.

TensorFlow Serving exposes, for a model called `mobilenet`:

    POST {host}/v1/models/mobilenet:predict     run inference
    GET  {host}/v1/models/mobilenet             model status

The predict endpoint takes {"instances": [...]} and answers
{"predictions": [[...]]}, one inner list of 1000 class probabilities per
submitted image.
"""
from __future__ import annotations

import json
import os

import httpx

from app.config import Settings


class TFServingError(Exception):
    """Raised when TensorFlow Serving is unreachable or returns an error."""


async def predict(instance: list, settings: Settings) -> list[float]:
    """Send one preprocessed image to TensorFlow Serving, return class scores."""
    url = f"{settings.tf_serving_url}/v1/models/{settings.model_name}:predict"

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(url, json={"instances": [instance]})
    except httpx.RequestError as exc:
        raise TFServingError(f"Could not reach TensorFlow Serving at {url}: {exc}") from exc

    if response.status_code != 200:
        raise TFServingError(
            f"TensorFlow Serving returned {response.status_code}: {response.text[:400]}"
        )

    payload = response.json()
    if "predictions" not in payload:
        raise TFServingError(f"Unexpected response from TensorFlow Serving: {payload}")

    return payload["predictions"][0]


async def model_status(settings: Settings) -> tuple[bool, str]:
    """Ask TensorFlow Serving whether the model is loaded and available."""
    url = f"{settings.tf_serving_url}/v1/models/{settings.model_name}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
    except httpx.RequestError as exc:
        return False, f"TensorFlow Serving unreachable: {exc}"

    if response.status_code != 200:
        return False, f"TensorFlow Serving returned {response.status_code}"

    versions = response.json().get("model_version_status", [])
    if versions and versions[0].get("state") == "AVAILABLE":
        return True, "Model is available"
    return False, f"Model not available yet: {versions}"


def load_labels(settings: Settings) -> dict[int, str]:
    """Load the index -> class name mapping written by the model-init step.

    Falls back to generic names so the API still answers (with unhelpful but
    valid labels) if the file is missing.
    """
    if not os.path.exists(settings.labels_path):
        return {}

    with open(settings.labels_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    return {int(key): value for key, value in raw.items()}
