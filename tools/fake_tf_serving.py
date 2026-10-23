"""A fake TensorFlow Serving server, for testing without the real thing.

Implements the two REST endpoints the backend uses, with the same request and
response shapes as TensorFlow Serving:

    GET  /v1/models/{model}             model status
    POST /v1/models/{model}:predict     inference

It returns deterministic fake scores, so it verifies the wiring (HTTP round
trip, payload shape, label lookup, ranking) without needing the real model
server. Used by verify.sh.

Run with:  python tools/fake_tf_serving.py [port]
"""
import hashlib
import sys

import numpy as np
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="Fake TensorFlow Serving")

NUM_CLASSES = 1000


@app.get("/v1/models/{model_name}")
async def model_status(model_name: str):
    return {
        "model_version_status": [
            {"version": "1", "state": "AVAILABLE", "status": {"error_code": "OK"}}
        ]
    }


@app.post("/v1/models/{model_name}:predict")
async def predict(model_name: str, request: Request):
    payload = await request.json()

    instances = payload.get("instances")
    if instances is None:
        return {"error": "missing 'instances' key"}

    array = np.array(instances, dtype=np.float32)
    # Mirror TensorFlow Serving: reject anything that is not (N, 224, 224, 3).
    if array.ndim != 4 or array.shape[1:] != (224, 224, 3):
        return {
            "error": f"expected shape (N, 224, 224, 3), got {array.shape}"
        }

    predictions = []
    for instance in array:
        # Deterministic winner derived from the image content, so repeated
        # runs on the same image give the same answer.
        digest = hashlib.sha256(instance.tobytes()).digest()
        winner = int.from_bytes(digest[:2], "big") % NUM_CLASSES

        scores = np.full(NUM_CLASSES, 0.00005, dtype=float)
        scores[winner] = 0.87
        scores[(winner + 1) % NUM_CLASSES] = 0.07
        scores[(winner + 2) % NUM_CLASSES] = 0.03
        predictions.append(scores.tolist())

    return {"predictions": predictions}


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
