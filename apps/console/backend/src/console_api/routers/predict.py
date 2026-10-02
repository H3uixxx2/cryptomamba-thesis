"""Predict screen endpoints: local checkpoint, optional remote API, offline replay."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from ..schemas.predict import CheckpointPredictionRequest, LivePredictRequest
from ..services import predict_service
from ._http import handle_domain_errors

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.get("")
@handle_domain_errors
def predict_setup() -> dict:
    return predict_service.build_setup()


@router.post("/checkpoint")
@handle_domain_errors
def predict_checkpoint(req: CheckpointPredictionRequest) -> dict:
    """Run a hash-verified local checkpoint over the submitted causal window."""
    return predict_service.run_checkpoint(req)


@router.post("/live")
@handle_domain_errors
def predict_live(req: LivePredictRequest) -> dict:
    """Run a prediction through the remote model API."""
    return predict_service.run_live(req)


@router.get("/offline")
@handle_domain_errors
def predict_offline(date: Optional[str] = None) -> dict:
    """Offline backup over the held-out test split. Always labelled offline."""
    return predict_service.run_offline(date)
