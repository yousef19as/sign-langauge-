"""Pydantic response schemas."""
from pydantic import BaseModel, Field


class ImagePredictionResponse(BaseModel):
    model_used: str
    prediction: dict


class ModelInfo(BaseModel):
    name: str
    path: str
    loaded: bool
    input_shape: list[int] | None = None
    num_classes: int | None = None
    classes: list[str] = []


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    models_total: int
    default_model: str
    confidence_threshold: float
    version: str
