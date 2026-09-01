"""Data screen HTTP endpoints.

Thin transport layer: parse the request, call ``services.data_service``, translate
domain errors. All data logic lives in the service (which in turn reuses the
vendored ``cryptomamba_ui`` package).
"""
from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile

from ..schemas.data import WindowRequest
from ..services import data_service
from ._http import handle_domain_errors

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("")
@handle_domain_errors
def get_data(mode: str = Query("paper", pattern="^(paper)$")) -> dict:
    """Load the paper dataset (the demo default). Upload uses POST /api/data/upload."""
    return data_service.load_paper_dataset()


@router.post("/window")
@handle_domain_errors
def build_data_window(req: WindowRequest) -> dict:
    """Build the model contract for one explicitly selected 14-candle window."""
    return data_service.build_explicit_window(
        candles=[candle.model_dump() for candle in req.candles],
        prediction_date=req.prediction_date,
        risk=req.risk,
    )


@router.post("/upload")
@handle_domain_errors
async def upload_data(file: UploadFile = File(...)) -> dict:
    """Validate + split an uploaded OHLCV CSV."""
    raw = await file.read(data_service.MAX_UPLOAD_BYTES + 1)
    return data_service.load_uploaded_dataset(raw, filename=file.filename)
