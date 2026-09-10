"""FastAPI application for room classification inference."""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from room_vlm.api.model_service import model_service
from room_vlm.api.schemas import HealthResponse, ModelInfoResponse, PredictResponse

logger = logging.getLogger(__name__)

ALLOWED_IMAGE = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/jpg",
}
ALLOWED_VIDEO = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/avi",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _max_bytes() -> int:
    return int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024


def _safe_suffix(filename: str | None, allowed: set[str]) -> str:
    name = Path(filename or "upload.bin").name
    # Sanitize: drop directory components already via Path.name
    suffix = Path(name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")
    return suffix


async def _save_upload(upload: UploadFile, allowed_mimes: set[str], allowed_exts: set[str]) -> Path:
    if upload.content_type and upload.content_type not in allowed_mimes:
        # Some browsers send odd MIME types; fall back to extension check.
        logger.warning("Unexpected content_type=%s for %s", upload.content_type, upload.filename)
    suffix = _safe_suffix(upload.filename, allowed_exts)
    max_bytes = _max_bytes()
    data = await upload.read()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="Upload exceeds maximum allowed size")
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy-load can be slow; load once at startup when not in test mode.
    if os.environ.get("ROOM_VLM_SKIP_MODEL_LOAD") != "1":
        try:
            model_service.load()
        except Exception:
            logger.exception("Model failed to load at startup")
            raise
    else:
        logger.warning("ROOM_VLM_SKIP_MODEL_LOAD=1 — model not loaded (test mode)")
    yield


app = FastAPI(title="Room VLM API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not model_service.loaded and os.environ.get("ROOM_VLM_SKIP_MODEL_LOAD") == "1":
        return ModelInfoResponse(
            model=model_service.model_name,
            adapter=model_service.adapter_name,
            device="unloaded",
            labels=list(model_service.info()["labels"]),
            strategy="candidate_scoring",
        )
    info = model_service.info()
    return ModelInfoResponse(**info)


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image_endpoint(file: UploadFile = File(...)) -> PredictResponse:
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    path = await _save_upload(file, ALLOWED_IMAGE, IMAGE_EXTS)
    try:
        result = model_service.predict_image_file(path)
        return PredictResponse(**result)
    finally:
        path.unlink(missing_ok=True)


@app.post("/predict/video", response_model=PredictResponse)
async def predict_video_endpoint(
    file: UploadFile = File(...),
    include_temporal: bool = Query(False),
) -> PredictResponse:
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    path = await _save_upload(file, ALLOWED_VIDEO, VIDEO_EXTS)
    try:
        result = model_service.predict_video_file(path, include_temporal=include_temporal)
        return PredictResponse(**result)
    finally:
        path.unlink(missing_ok=True)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    logger.exception("Unhandled error")
    # Never leak filesystem paths or internals.
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
