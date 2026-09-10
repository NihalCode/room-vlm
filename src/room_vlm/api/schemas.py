"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ModelInfoResponse(BaseModel):
    model: str
    adapter: str | None = None
    device: str
    labels: list[str]
    strategy: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]
    model: str
    adapter: str | None = None
    latency_ms: float
    abstained: bool = False
    frames_used: int | None = None
    temporal_predictions: list[dict[str, Any]] | None = None
