"""Streamlit frontend: upload an image, get a prediction.

Communicates only with the FastAPI backend, never with TensorFlow Serving
directly. The backend address arrives through APP_BACKEND_URL, which Compose
sets to the backend's service name on the shared network.
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("APP_BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Image Classification", layout="centered")

st.title("Image Classification")
st.caption("MobileNetV2 served by TensorFlow Serving")


def get_model_status() -> tuple[bool, str]:
    try:
        response = requests.get(f"{BACKEND_URL}/model/status", timeout=5)
        if response.status_code != 200:
            return False, f"Backend returned {response.status_code}"
        payload = response.json()
        return bool(payload.get("available")), payload.get("detail") or ""
    except requests.RequestException as exc:
        return False, f"Backend unreachable: {exc}"


with st.sidebar:
    st.subheader("Service status")
    available, detail = get_model_status()
    if available:
        st.success("Model server ready")
    else:
        st.warning("Model not ready")
        st.caption(detail)
    st.caption(f"Backend: {BACKEND_URL}")
    if st.button("Refresh status"):
        st.rerun()


uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="The image is sent to the backend, which runs inference on TensorFlow Serving.",
)

if uploaded is not None:
    st.image(uploaded, caption=uploaded.name, use_column_width=True)

    if st.button("Classify", type="primary"):
        with st.spinner("Running inference..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
                st.stop()

        if response.status_code == 503:
            st.error(
                "The model server is not available. "
                "If you just started the stack, wait for model-init to finish and retry."
            )
            st.caption(response.json().get("detail", ""))
            st.stop()
        if response.status_code != 200:
            st.error(f"Backend error {response.status_code}")
            st.caption(response.text[:500])
            st.stop()

        payload = response.json()
        predictions = payload["predictions"]

        st.subheader("Prediction")
        best = predictions[0]
        st.metric(
            label=best["label"].replace("_", " "),
            value=f"{best['probability'] * 100:.1f}%",
        )

        st.caption(f"Inference took {payload['inference_time_ms']} ms")

        st.subheader("Top 5")
        for item in predictions:
            st.write(f"{item['label'].replace('_', ' ')}  -  {item['probability'] * 100:.2f}%")
            st.progress(min(float(item["probability"]), 1.0))
else:
    st.info("Upload an image to get started.")
