"""Export MobileNetV2 as a SavedModel for TensorFlow Serving.

Runs once, as the `model-init` service in docker-compose.yml, and writes
into the shared `model-store` volume:

    /models/mobilenet/1/        the SavedModel TensorFlow Serving loads
    /models/labels.json         index -> human readable ImageNet class name

TensorFlow is only needed here. The backend talks to TensorFlow Serving over
REST, so it stays a small image without TensorFlow installed.
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_NAME = os.getenv("MODEL_NAME", "mobilenet")
MODEL_VERSION = "1"


def export_saved_model() -> str:
    """Download MobileNetV2 (ImageNet weights) and export it for Serving."""
    target = os.path.join(MODEL_DIR, MODEL_NAME, MODEL_VERSION)

    # TensorFlow Serving picks the highest numbered version directory, so an
    # already exported model means there is nothing to do.
    if os.path.exists(os.path.join(target, "saved_model.pb")):
        print(f"[model-init] SavedModel already present at {target}, skipping export")
        return target

    print("[model-init] Downloading MobileNetV2 with ImageNet weights...")
    model = MobileNetV2(weights="imagenet")

    os.makedirs(target, exist_ok=True)
    print(f"[model-init] Exporting SavedModel to {target} ...")
    # Keras 3 exports a serving-ready SavedModel through .export().
    model.export(target)
    print("[model-init] Export finished")
    return target


def export_labels() -> str:
    """Write the index -> class name mapping used by the backend."""
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    if os.path.exists(labels_path):
        print(f"[model-init] {labels_path} already present, skipping")
        return labels_path

    # One one-hot row per class: decode_predictions then reports, for every
    # row, the name of exactly that class. This uses only the public Keras
    # API instead of depending on the location of the class-index file.
    identity = np.eye(1000, dtype="float32")
    decoded = decode_predictions(identity, top=1)
    labels = {str(index): entry[0][1] for index, entry in enumerate(decoded)}

    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)

    print(f"[model-init] Wrote {len(labels)} labels to {labels_path}")
    return labels_path


def install_models_config() -> str:
    """Copy models.config into the shared volume for TensorFlow Serving.

    TensorFlow Serving is started with --model_config_file, the approach shown
    in class, rather than the simpler --model_name / --model_base_path flags.
    """
    destination = os.path.join(MODEL_DIR, "models.config")
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.config")

    with open(source, encoding="utf-8") as src, open(destination, "w", encoding="utf-8") as dst:
        dst.write(src.read())

    print(f"[model-init] Installed model config at {destination}")
    return destination


def main() -> None:
    print(f"[model-init] TensorFlow {tf.__version__}")
    target = export_saved_model()
    export_labels()
    install_models_config()

    contents = sorted(os.listdir(target))
    print(f"[model-init] Contents of {target}: {contents}")
    if "saved_model.pb" not in contents:
        raise SystemExit("[model-init] ERROR: saved_model.pb was not created")
    print("[model-init] Done. TensorFlow Serving can now load the model.")


if __name__ == "__main__":
    main()
