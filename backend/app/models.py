"""Response schemas for the prediction API."""
from pydantic import BaseModel, ConfigDict, Field

# This app uses fields named model_* (model_name), which collide with
# Pydantic's reserved "model_" prefix unless the namespace is opened up.
ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class Prediction(BaseModel):
    label: str = Field(..., description="Human readable ImageNet class name")
    class_index: int = Field(..., description="Index of the class in the model output")
    probability: float = Field(..., description="Softmax probability, 0..1")


class PredictionResponse(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    filename: str
    model_name: str
    predictions: list[Prediction]
    inference_time_ms: float


class ModelStatus(BaseModel):
    model_config = ALLOW_MODEL_PREFIX

    model_name: str
    available: bool
    detail: str | None = None
