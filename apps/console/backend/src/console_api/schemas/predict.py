"""Request models for the Predict screen."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LivePredictRequest(BaseModel):
    api_url: str = Field(..., min_length=1)
    prediction_date: Optional[str] = None
    risk: float = Field(2.0, ge=0.5, le=10.0)


class PredictCandle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class CheckpointPredictionRequest(BaseModel):
    model_id: Literal["cmamba_v_reproduced", "s5_full"]
    candles: list[PredictCandle] = Field(..., min_length=1, max_length=10_000)
    prediction_date: str
