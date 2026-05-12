"""Predictor – supports arabic and asl models."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from arabic_map import ARABIC_CLASS_NAMES, ARABIC_LETTERS_MAP, ASL_CLASS_NAMES
from config import settings
from schemas import ImagePredictionResponse, ModelInfo

logger = logging.getLogger(__name__)

_registry: dict[str, object] = {}


def _class_names(model_name: str) -> list[str]:
    return ASL_CLASS_NAMES if model_name == "asl" else ARABIC_CLASS_NAMES


def _image_size(model_name: str) -> int:
    return settings.image_size_per_model.get(model_name, 64)


def load_all_models() -> None:
    import tensorflow as tf
    for name, path_str in settings.model_registry.items():
        path = Path(path_str)
        if not path.exists():
            logger.warning("Model '%s' not found at %s – skipping.", name, path)
            _registry[name] = None
            continue
        try:
            logger.info("Loading model '%s' from %s …", name, path)
            _registry[name] = tf.keras.models.load_model(str(path))
            m = _registry[name]
            logger.info("Model '%s' ready – input: %s, classes: %d", name, m.input_shape, m.output_shape[-1])
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", name, exc)
            _registry[name] = None


def unload_all_models() -> None:
    _registry.clear()


def loaded_count() -> int:
    return sum(1 for m in _registry.values() if m is not None)


def total_count() -> int:
    return len(settings.model_registry)


def list_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            name=name,
            path=path_str,
            loaded=_registry.get(name) is not None,
            input_shape=list(_registry[name].input_shape[1:]) if _registry.get(name) else None,
            num_classes=int(_registry[name].output_shape[-1]) if _registry.get(name) else None,
            classes=_class_names(name),
        )
        for name, path_str in settings.model_registry.items()
    ]


def _get_model(name: str):
    if name not in settings.model_registry:
        raise KeyError(f"Unknown model '{name}'. Available: {list(settings.model_registry.keys())}")
    m = _registry.get(name)
    if m is None:
        raise RuntimeError(f"Model '{name}' failed to load. Check server logs.")
    return m


def _preprocess(img: Image.Image, size: int) -> np.ndarray:
    arr = np.array(img.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(image_bytes: bytes, model_name: str) -> ImagePredictionResponse:
    model = _get_model(model_name)
    classes = _class_names(model_name)
    size = _image_size(model_name)

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    probs: np.ndarray = model.predict(_preprocess(img, size), verbose=0)[0]
    best_idx = int(np.argmax(probs))
    label = classes[best_idx]
    confidence = round(float(probs[best_idx]), 4)

    prediction = {
        "label": label,
        "confidence": confidence,
        "above_threshold": confidence >= settings.confidence_threshold,
    }

    # arabic model adds the Arabic character
    if model_name == "arabic":
        prediction["arabic"] = ARABIC_LETTERS_MAP.get(label, "?")

    return ImagePredictionResponse(
        model_used=model_name,
        prediction=prediction,
    )
