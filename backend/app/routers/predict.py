from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.models import ModelStatus, Prediction, PredictionResponse
from app.services.preprocessing import InvalidImageError, preprocess
from app.services.tfserving import TFServingError, load_labels, model_status, predict

router = APIRouter(tags=["inference"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
TOP_K = 5


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(..., description="Image to classify"),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """Classify an uploaded image using the model served by TensorFlow Serving."""
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=422, detail="The uploaded file is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    try:
        instance = preprocess(image_bytes, image_size=settings.image_size)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    started = time.perf_counter()
    try:
        scores = await predict(instance, settings)
    except TFServingError as exc:
        # 503: the backend is fine, the model server is not.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    labels = load_labels(settings)

    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:TOP_K]
    predictions = [
        Prediction(
            label=labels.get(index, f"class_{index}"),
            class_index=index,
            probability=float(score),
        )
        for index, score in ranked
    ]

    return PredictionResponse(
        filename=file.filename or "upload",
        model_name=settings.model_name,
        predictions=predictions,
        inference_time_ms=round(elapsed_ms, 2),
    )


@router.get("/model/status", response_model=ModelStatus)
async def get_model_status(settings: Settings = Depends(get_settings)) -> ModelStatus:
    """Report whether TensorFlow Serving has the model loaded."""
    available, detail = await model_status(settings)
    return ModelStatus(model_name=settings.model_name, available=available, detail=detail)
