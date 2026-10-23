"""Backend settings, read from environment variables."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() because this app legitimately uses fields named
    # model_* (model_name), which would otherwise collide with Pydantic's
    # reserved "model_" prefix and emit warnings.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=()
    )

    # Base URL of the TensorFlow Serving REST API. Compose sets this to the
    # service name, so it resolves over the internal Docker network.
    tf_serving_url: str = "http://tf-serving:8501"
    model_name: str = "mobilenet"

    # Written by the model-init container into the shared volume.
    labels_path: str = "/models/labels.json"

    # MobileNetV2 input size.
    image_size: int = 224

    request_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
