"""Streamlit frontend.

Upload an image, send it to TensorFlow Serving, show the predictions.
There is no backend: this file talks to the model server directly.
"""
import io
import json
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image

# Compose sets this to the tf-serving service name, so Docker's internal DNS
# resolves it. The /v1/models/<name>:predict URL is the TensorFlow Serving
# REST API.
TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://tf-serving:8501")
MODEL_NAME = "mobilenet"
PREDICT_URL = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}:predict"
STATUS_URL = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}"
LABELS_PATH = "/models/labels.json"

st.set_page_config(page_title="Image Classification")
st.title("Image Classification")
st.caption("MobileNetV2 served by TensorFlow Serving")


def load_labels():
    """Read the number -> class name mapping written by model-init."""
    if not os.path.exists(LABELS_PATH):
        return {}
    with open(LABELS_PATH) as f:
        return json.load(f)


def preprocess(image_bytes):
    """Resize to 224x224 and scale pixels to the range MobileNetV2 expects.

    MobileNetV2 was trained on pixels scaled from [0, 255] to [-1, 1],
    which is what the division below does.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return array.tolist()


def model_is_ready():
    try:
        response = requests.get(STATUS_URL, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Sidebar: is the model server up?
with st.sidebar:
    st.subheader("Model server")
    if model_is_ready():
        st.success("Ready")
    else:
        st.warning("Not ready yet. If you just started the app, wait for "
                   "model-init to finish and press the button below.")
    if st.button("Check again"):
        st.rerun()


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("Upload an image to get started.")
else:
    st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    if st.button("Classify", type="primary"):
        instance = preprocess(uploaded_file.getvalue())

        # TensorFlow Serving expects {"instances": [...]} and answers
        # {"predictions": [[...1000 scores...]]}.
        try:
            response = requests.post(
                PREDICT_URL, json={"instances": [instance]}, timeout=60
            )
        except requests.RequestException as error:
            st.error(f"Could not reach the model server: {error}")
            st.stop()

        if response.status_code != 200:
            st.error(f"Model server error {response.status_code}")
            st.caption(response.text[:300])
            st.stop()

        scores = response.json()["predictions"][0]
        labels = load_labels()

        # Sort the 1000 scores and keep the 5 highest.
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        top_5 = ranked[:5]

        best_index, best_score = top_5[0]
        best_label = labels.get(str(best_index), f"class {best_index}")

        st.subheader("Prediction")
        st.metric(best_label.replace("_", " "), f"{best_score * 100:.1f}%")

        st.subheader("Top 5")
        for index, score in top_5:
            name = labels.get(str(index), f"class {index}").replace("_", " ")
            st.write(f"{name} - {score * 100:.2f}%")
            st.progress(float(score))
