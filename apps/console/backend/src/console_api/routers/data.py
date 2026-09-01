"""Data screen API — mirrors the Streamlit Data page, reusing cryptomamba_ui logic.

No data logic lives here: schema validation, daily aggregation, splitting, the
14-day window and the tensor preview all come from the reused package.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .. import logic

router = APIRouter(prefix="/api/data", tags=["data"])

# Daily BTC OHLCV is a few hundred KB; cap uploads so a pathological file can't
# OOM the single worker (the read is otherwise unbounded).
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_SPLIT_COLOR = {
    "train": "var(--gray-200)",
    "validation": "var(--signal-blue)",
    "test": "var(--orange-500)",
    "out_of_scope": "var(--text-muted)",
}


def fig_json(fig: Any) -> dict:
    """Serialize a Plotly figure to a JSON-safe dict for Plotly.js on the client."""
    return json.loads(fig.to_json())


def _candle_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the explicit, JSON-safe candle contract consumed by the explorer."""
    return [
        {
            "date": str(row.date),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "split": str(row.dataset_split),
        }
        for row in frame.itertuples(index=False)
    ]


class WindowCandle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    split: str


class WindowRequest(BaseModel):
    candles: list[WindowCandle]
    prediction_date: str
    risk: float = Field(2.0, ge=0.5, le=10.0)


def _tensor_contract(window_df: pd.DataFrame) -> dict:
    _, tensor_payload = logic.data.model_tensor_preview(window_df)
    return {
        "shape": tensor_payload["tensor_shape_sent_to_model"],
        "feature_order": tensor_payload["feature_order"],
        "seq_len": tensor_payload["seq_len"],
        "volume_rule": tensor_payload["volume_rule"],
        "model_config": tensor_payload["model_config"],
    }


def _model_window_contract(
    processed: pd.DataFrame,
    *,
    prediction_date: str | None,
    risk: float,
) -> dict:
    prediction_payload = logic.data.build_predict_payload(
        processed,
        prediction_date=prediction_date,
        risk=risk,
    )
    target_date = str(prediction_payload["prediction_date"])
    window_df = logic.data.select_window(processed, target_date)

    split_by_date = processed.drop_duplicates("date").set_index("date")["dataset_split"]
    window_df["dataset_split"] = window_df["date"].map(split_by_date).fillna("unknown")
    window_rows = _candle_rows(window_df)
    return {
        "window": {
            "start": str(window_df["date"].iloc[0]),
            "end": str(window_df["date"].iloc[-1]),
            "size": logic.data.MODEL_WINDOW_SIZE,
            "rows": window_rows,
        },
        "tensor": _tensor_contract(window_df),
        "prediction_payload": prediction_payload,
    }


def _build_response(bundle: "logic.DatasetBundle") -> dict:
    processed = bundle.processed_df
    if processed is None or processed.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty after processing.")
    processed = processed.sort_values("date").reset_index(drop=True)

    summary = logic.data.split_summary(processed)
    splits = [
        {
            "split": row["split"],
            "rows": int(row["rows"]),
            "from": str(row["from"]),
            "to": str(row["to"]),
        }
        for _, row in summary.iterrows()
    ]
    split_counts = {s["split"]: s for s in splits}

    # Metric strip: total + each paper/chronological split (data-driven, 4 cards).
    total_rows = int(processed.shape[0])
    date_from = str(processed["date"].iloc[0])
    date_to = str(processed["date"].iloc[-1])
    metrics: list[dict] = [
        {
            "label": "Total candles",
            "value": f"{total_rows:,}",
            "unit": "days",
            "range": f"{date_from} → {date_to}",
            "color": "var(--text-primary)",
        }
    ]
    for split_key, label in (("train", "Train"), ("validation", "Validation"), ("test", "Test")):
        s = split_counts.get(split_key)
        if s is None:
            continue
        metrics.append(
            {
                "label": label,
                "value": f"{s['rows']:,}",
                "unit": "days",
                "range": f"{s['from']} → {s['to']}",
                "color": _SPLIT_COLOR.get(split_key, "var(--text-primary)"),
            }
        )

    # Default remains the latest 14-day input window and next-calendar-day target.
    candles = _candle_rows(processed)
    model_contract = _model_window_contract(
        processed,
        prediction_date=None,
        risk=2.0,
    )

    # Pipeline stages — reflect the real steps the bundle went through (all PASS).
    pipeline = [
        {
            "stage": "Read and validate OHLCV schema",
            "artifact": f"{len(bundle.raw_df):,} raw rows",
            "status": "PASS",
        },
        {
            "stage": "Aggregate daily candles",
            "artifact": f"{len(bundle.daily_df):,} daily candles",
            "status": "PASS",
        },
        {
            "stage": f"Create splits ({bundle.split_strategy})",
            "artifact": f"{total_rows:,} processed rows · {len(splits)} splits",
            "status": "PASS",
        },
        {"stage": "Build 14-day input window", "artifact": "tensor [1, 6, 14]", "status": "PASS"},
        {
            "stage": "Volume scaling",
            "artifact": model_contract["tensor"]["volume_rule"],
            "status": "PASS",
        },
    ]

    return {
        "source": {
            "label": bundle.source_label,
            "detail": bundle.source_detail,
            "strategy": bundle.split_strategy,
        },
        "metrics": metrics,
        "splits": splits,
        "candles": candles,
        "processing": {
            "raw_rows": len(bundle.raw_df),
            "daily_rows": len(bundle.daily_df),
            "processed_rows": total_rows,
            "split_strategy": bundle.split_strategy,
        },
        **model_contract,
        "pipeline": pipeline,
    }


@router.get("")
def get_data(mode: str = Query("paper", pattern="^(paper)$")) -> dict:
    """Load the paper dataset (the demo default). Upload uses POST /api/data/upload."""
    bundle = logic.DATASET_SERVICE.load_paper_sample()
    return _build_response(bundle)


@router.post("/window")
def build_data_window(req: WindowRequest) -> dict:
    """Build the model contract for one explicitly selected 14-candle window."""
    if len(req.candles) != logic.data.MODEL_WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Window must contain exactly {logic.data.MODEL_WINDOW_SIZE} candles.",
        )

    input_rows = [candle.model_dump() for candle in req.candles]
    input_dates = [row["date"] for row in input_rows]
    if len(set(input_dates)) != logic.data.MODEL_WINDOW_SIZE:
        raise HTTPException(status_code=400, detail="Window candle dates must be unique.")

    target = pd.to_datetime(req.prediction_date, format="%Y-%m-%d", errors="coerce")
    if pd.isna(target) or target.strftime("%Y-%m-%d") != req.prediction_date:
        raise HTTPException(status_code=400, detail="prediction_date must use YYYY-MM-DD.")

    frame = pd.DataFrame(input_rows).rename(columns={"split": "dataset_split"})
    try:
        normalized = logic.data.normalize_candles(frame)
    except logic.CandleDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_dates = normalized["date"].astype(str).tolist()
    if input_dates != normalized_dates:
        raise HTTPException(
            status_code=400,
            detail="Window candles must use unique YYYY-MM-DD dates in chronological order.",
        )
    if any(pd.to_datetime(day) >= target for day in normalized_dates):
        raise HTTPException(
            status_code=400,
            detail="All window candle dates must be strictly before prediction_date.",
        )

    normalized["dataset_split"] = [row["dataset_split"] for row in frame.to_dict("records")]
    try:
        return _model_window_contract(
            normalized,
            prediction_date=req.prediction_date,
            risk=req.risk,
        )
    except logic.CandleDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)) -> dict:
    """Validate + split an uploaded OHLCV CSV using the reused chronological splitter."""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV is too large (maximum {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    buffer = io.BytesIO(raw)
    try:
        bundle = logic.DATASET_SERVICE.load_uploaded_csv(buffer, filename=file.filename)
    except logic.CandleDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (
        csv.Error,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        OSError,
    ) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc
    return _build_response(bundle)
