"""
Sign Language Recognition API  v2.1
=====================================
POST /predict/arabic   – Arabic Sign Language (32 classes, 64×64)
POST /predict/english  – ASL English letters  (29 classes, 32×32)
GET  /models           – list all models and their status
GET  /health           – server status
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

import predictor
from config import settings
from schemas import HealthResponse, ImagePredictionResponse, ModelsResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_all_models()
    yield
    predictor.unload_all_models()


app = FastAPI(
    title="Sign Language Recognition API",
    version="2.1.0",
    description=(
        "Two dedicated endpoints:\n\n"
        "- **POST /predict/arabic** — Arabic Sign Language, 32 classes, returns Arabic character (ع، ب، ت...)\n"
        "- **POST /predict/english** — ASL English letters, 29 classes, returns English letter (A, B, C...)\n"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate(file: UploadFile, content: bytes) -> None:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type '{file.content_type}'. Allowed: jpeg, png, webp, bmp",
        )
    if len(content) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max: {settings.max_image_bytes // (1024*1024)} MB.",
        )


def _handle_errors(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc))
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail=str(exc))


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    """Server and model status."""
    return HealthResponse(
        status="ok" if predictor.loaded_count() > 0 else "degraded",
        models_loaded=predictor.loaded_count(),
        models_total=predictor.total_count(),
        default_model=settings.default_model,
        confidence_threshold=settings.confidence_threshold,
        version="2.1.0",
    )


@app.get("/models", response_model=ModelsResponse, tags=["Status"])
async def list_models():
    """List all registered models with their classes and load status."""
    return ModelsResponse(
        default_model=settings.default_model,
        models=predictor.list_models(),
    )


@app.post(
    "/predict/arabic",
    response_model=ImagePredictionResponse,
    tags=["Prediction"],
    summary="Classify Arabic sign language image",
)
async def predict_arabic(
    file: UploadFile = File(..., description="JPEG/PNG image of an Arabic hand sign"),
):
    """
    Upload a hand-sign image → get predicted **Arabic letter** + confidence + top-5.

    Returns Arabic character (e.g. **ع**, **ب**, **ت**...)
    """
    content = await file.read()
    _validate(file, content)
    try:
        return predictor.predict_image(content, "arabic")
    except Exception as exc:
        _handle_errors(exc)


@app.post(
    "/predict/english",
    response_model=ImagePredictionResponse,
    tags=["Prediction"],
    summary="Classify ASL English sign language image",
)
async def predict_english(
    file: UploadFile = File(..., description="JPEG/PNG image of an ASL hand sign"),
):
    """
    Upload a hand-sign image → get predicted **English letter** + confidence + top-5.

    Returns English letter (e.g. **A**, **B**, **C**...) + del / nothing / space
    """
    content = await file.read()
    _validate(file, content)
    try:
        return predictor.predict_image(content, "asl")
    except Exception as exc:
        _handle_errors(exc)
