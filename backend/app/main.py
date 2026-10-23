from fastapi import FastAPI

from app.routers.predict import router as predict_router

app = FastAPI(
    title="Image Classification API",
    description=(
        "FastAPI backend that accepts an uploaded image, preprocesses it and "
        "runs inference against a MobileNetV2 model served by TensorFlow Serving."
    ),
    version="1.0.0",
)

app.include_router(predict_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Liveness probe for the backend itself (does not touch the model server)."""
    return {"status": "ok"}
