"""Download MobileNetV2 and export it for TensorFlow Serving.

This container runs once, writes the model into the shared volume, and exits.
TensorFlow is only needed here: the frontend talks to TensorFlow Serving over
HTTP, so it does not need TensorFlow installed.

It writes two things into /models:
  mobilenet/1/     the model, in the format TensorFlow Serving expects
  labels.json      maps the model's output numbers to class names
"""
import json
import os

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions

MODEL_DIR = "/models"
TARGET = os.path.join(MODEL_DIR, "mobilenet", "1")
LABELS = os.path.join(MODEL_DIR, "labels.json")


def export_model():
    if os.path.exists(os.path.join(TARGET, "saved_model.pb")):
        print("Model already exported, skipping.")
        return

    print("Downloading MobileNetV2...")
    model = MobileNetV2(weights="imagenet")

    print(f"Exporting model to {TARGET} ...")
    os.makedirs(TARGET, exist_ok=True)
    model.export(TARGET)
    print("Model exported.")


def export_labels():
    if os.path.exists(LABELS):
        print("Labels already exported, skipping.")
        return

    # The model outputs 1000 numbers, one score per class. To find out which
    # class each position means, we feed decode_predictions one row per class
    # with a 1 in that position and read back the name it reports.
    one_hot_rows = np.eye(1000, dtype="float32")
    decoded = decode_predictions(one_hot_rows, top=1)
    labels = {str(i): entry[0][1] for i, entry in enumerate(decoded)}

    with open(LABELS, "w") as f:
        json.dump(labels, f)
    print(f"Wrote {len(labels)} labels to {LABELS}")


export_model()
export_labels()
print("Done. TensorFlow Serving can now load the model.")
