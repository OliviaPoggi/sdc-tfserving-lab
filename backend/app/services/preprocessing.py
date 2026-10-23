"""Image preprocessing for MobileNetV2.

Done with Pillow and NumPy instead of TensorFlow, so the backend image does
not need to carry TensorFlow just to resize an image. The arithmetic matches
`tf.keras.applications.mobilenet_v2.preprocess_input`, which scales pixels
from [0, 255] to [-1, 1].
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImageError(Exception):
    """Raised when the uploaded bytes are not a readable image."""


def preprocess(image_bytes: bytes, image_size: int = 224) -> list[list[list[float]]]:
    """Turn raw uploaded bytes into the nested list TF Serving expects.

    Returns a (image_size, image_size, 3) nested list of floats in [-1, 1].
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Could not read the uploaded file as an image: {exc}") from exc

    # Flatten transparency and greyscale onto three channels.
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((image_size, image_size), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32)
    array = array / 127.5 - 1.0

    return array.tolist()
