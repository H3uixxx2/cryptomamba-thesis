"""Predict screen business logic.

Honesty rules preserved: live results are labelled ``live``, the frozen backup is
forced to ``offline``, and out-of-distribution windows (after the paper horizon)
are flagged. No model inference is implemented here — the checkpoint path shells
out to the model-backend worker, live calls the remote API, offline reads the
frozen artifact.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from ..core import config
from ..core.errors import (
    ConsoleError,
    EvidenceUnavailableError,
    InvalidInputError,
    UpstreamError,
)
from .. import logic
from ..loaders.checkpoint_inference import (
    CheckpointInferenceError,
    run_checkpoint_inference,
)
from ..loaders.final_evidence import EvidenceIntegrityError, FinalEvidence
from ..schemas.predict import CheckpointPredictionRequest, LivePredictRequest
from .data_service import fig_json

def _prediction_rows(path, result_type: str) -> "pd.DataFrame | None":
    """Load one validated prediction series from an artifact CSV."""
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return None
    needed = {"result_type", "split", "prediction_date", "current_close", "target_close", "predicted_close"}
    if not needed.issubset(frame.columns):
        return None
    selected = frame[
        frame["result_type"].eq(result_type) & frame["split"].eq("test")
    ].copy()
    if selected.empty:
        return None
    selected["prediction_date"] = selected["prediction_date"].astype(str)
    return selected.sort_values("prediction_date")


def _forecast_test_df() -> "pd.DataFrame | None":
    """Frozen official CryptoMamba-v predictions on the paper test split."""
    return _prediction_rows(
        config.EVALUATION_DIR / "forecast_predictions.csv",
        "official_checkpoint",
    )


def _forecast_variant(*, variant_id: str, label: str, source: str, row) -> dict:
    current = float(row["current_close"])
    predicted = float(row["predicted_close"])
    actual = float(row["target_close"])
    return {
        "id": variant_id,
        "label": label,
        "source": source,
        "prediction_date": str(row["prediction_date"]),
        "predicted_close": predicted,
        "move_pct": logic.trading_logic.pct(predicted, current),
        "error_pct": (abs(predicted - actual) / actual * 100.0) if actual else None,
        "direction_correct": bool((predicted > current) == (actual > current)),
    }


def _offline_variants(raw_row) -> list[dict]:
    """The frozen offline forecast, as the single artifact-backed variant."""
    return [
        _forecast_variant(
            variant_id="raw",
            label="Raw CryptoMamba-v",
            source="official_checkpoint",
            row=raw_row,
        )
    ]











def _checkpoint_models_from_evidence() -> dict:
    evidence = FinalEvidence(config.FINAL_EVIDENCE_DIR)
    provenance = evidence.read_json("model/checkpoint_provenance.json")
    models = provenance.get("models") if isinstance(provenance, dict) else None
    if not isinstance(models, dict):
        raise EvidenceIntegrityError("checkpoint provenance has no models object")
    return models


def _verified_checkpoint_model(models: dict, model_id: str) -> dict:
    model = models.get(model_id)
    if not isinstance(model, dict):
        raise EvidenceIntegrityError(f"checkpoint provenance is missing {model_id}")
    window_days = model.get("window_days")
    if (
        model.get("hash_verified") is not True
        or isinstance(window_days, bool)
        or not isinstance(window_days, int)
        or window_days <= 0
        or not isinstance(model.get("display_name"), str)
        or not isinstance(model.get("checkpoint_sha256"), str)
        or not isinstance(model.get("source_commit"), str)
    ):
        raise EvidenceIntegrityError(
            f"checkpoint provenance is invalid for {model_id}"
        )
    return model


def _iso_date(value: str, *, field: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise InvalidInputError(f"{field} must use valid YYYY-MM-DD format.") from exc
    if parsed.isoformat() != value:
        raise InvalidInputError(f"{field} must use YYYY-MM-DD format.")
    return parsed


def _checkpoint_window_rows(
    req: CheckpointPredictionRequest, *, window_days: int
) -> list[dict]:
    prediction_date = _iso_date(req.prediction_date, field="prediction_date")
    dated_candles = [
        (candle, _iso_date(candle.date, field=f"candles[{index}].date"))
        for index, candle in enumerate(req.candles)
    ]
    if any(
        current <= previous
        for (_, previous), (_, current) in zip(dated_candles, dated_candles[1:])
    ):
        raise InvalidInputError("Candle dates must be unique and strictly increasing.")

    available = [item for item in dated_candles if item[1] < prediction_date]
    if len(available) < window_days:
        raise InvalidInputError((
                f"{req.model_id} requires at least {window_days} history candles "
                f"strictly before prediction_date; received {len(available)}."
            ),
        )
    selected = available[-window_days:]
    if any(
        current - previous != timedelta(days=1)
        for (_, previous), (_, current) in zip(selected, selected[1:])
    ):
        raise InvalidInputError("Selected candle dates must be contiguous daily.")
    if prediction_date - selected[-1][1] != timedelta(days=1):
        raise InvalidInputError((
                "prediction_date must be exactly one calendar day after the "
                "final selected candle."),
        )
    return [candle.model_dump(mode="json") for candle, _ in selected]


def _paper_df():
    return logic.DATASET_SERVICE.load_paper_sample().processed_df


def _result_from_prediction(prediction: dict, window_df, *, ood: bool, provenance: dict | None) -> dict:
    """Assemble the unified result the frontend renders (live + offline share this)."""
    last_close = float(prediction["last_close"])
    predicted_close = float(prediction["predicted_close"])
    move_pct = logic.trading_logic.pct(predicted_close, last_close)
    candle_fig = logic.charts.candle_chart(
        window_df,
        f"{len(window_df)}-day input",
        prediction={
            "prediction_date": prediction.get("prediction_date"),
            "last_close": last_close,
            "predicted_close": predicted_close,
        },
        height=320,
    )
    result = {
        "inference_type": prediction.get("inference_type"),
        "model_id": prediction.get("model_id", "cmamba_v"),
        "prediction_date": prediction.get("prediction_date"),
        "last_close": last_close,
        "predicted_close": predicted_close,
        "move_pct": move_pct,
        "vanilla_action": prediction.get("vanilla_action"),
        "smart_action": prediction.get("smart_action"),
        "smart_pct": prediction.get("smart_pct"),
        "checkpoint_sha256": prediction.get("checkpoint_sha256"),
        "source_commit": prediction.get("source_commit"),
        "ood": ood,
        "window": {
            "start": str(window_df["date"].iloc[0]),
            "end": str(window_df["date"].iloc[-1]),
            "size": int(window_df.shape[0]),
            "rows": [
                {
                    "date": str(row.date),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                }
                for row in window_df.itertuples(index=False)
            ],
        },
        "charts": {"candle": fig_json(candle_fig)},
    }
    if provenance is not None:
        result["provenance"] = provenance
    return result


def _offline_by_date(date_str: str) -> dict | None:
    """Build an offline result for a historical test date from frozen predictions.
    Because the date is in the past, the actual outcome is known and returned too."""
    ft = _forecast_test_df()
    if ft is None:
        return None
    dates = ft["prediction_date"].tolist()
    if date_str not in dates:
        # Snap any out-of-range / malformed input to the nearest real test date
        # rather than silently serving an unrelated fixture.
        target = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(target):
            date_str = dates[0]
        else:
            diffs = (pd.to_datetime(pd.Series(dates)) - target).abs()
            date_str = dates[int(diffs.values.argmin())]
    m = ft[ft["prediction_date"] == date_str]
    if m.empty:
        return None
    row = m.iloc[0]
    current = float(row["current_close"])
    predicted = float(row["predicted_close"])
    actual = float(row["target_close"])
    try:
        window = logic.data.select_window(_paper_df(), date_str)
    except logic.CandleDataError:
        return None
    vanilla = logic.trading_logic.vanilla_signal(current, predicted)
    smart_action, smart_pct = logic.trading_logic.smart_signal(current, predicted, 2.0)
    prediction = {
        "inference_type": "offline",
        "model_id": "cmamba_v",
        "prediction_date": date_str,
        "last_close": current,
        "predicted_close": predicted,
        "vanilla_action": vanilla.upper(),
        "smart_action": smart_action.upper(),
        "smart_pct": smart_pct,
        "source_commit": str(row.get("source_commit") or ""),
    }
    ood = bool(logic.data.is_out_of_distribution(window))
    result = _result_from_prediction(prediction, window, ood=ood, provenance=None)
    result["available"] = True
    # Historical date → actual is known; expose the predicted-vs-actual comparison.
    result["actual_close"] = actual
    result["error_pct"] = (abs(predicted - actual) / actual * 100.0) if actual else None
    result["actual_return_pct"] = ((actual - current) / current * 100.0) if current else None
    result["direction_correct"] = bool((predicted > current) == (actual > current))
    result["forecast_variants"] = _offline_variants(row)
    return result


def build_setup() -> dict:
    """Setup metadata for the Predict screen: valid date range + defaults."""
    df = _paper_df()
    min_date, max_date = logic.data.prediction_date_bounds(df)
    last_close = float(df["close"].iloc[-1])
    ft = _forecast_test_df()
    offline_dates = ft["prediction_date"].tolist() if ft is not None else []
    checkpoint_models = []
    checkpoint_error = None
    try:
        models = _checkpoint_models_from_evidence()
        for model_id in ("cmamba_v_reproduced", "s5_full"):
            model = _verified_checkpoint_model(models, model_id)
            checkpoint_models.append(
                {
                    "id": model_id,
                    "label": model["display_name"],
                    "window_days": int(model["window_days"]),
                    "checkpoint_sha256": model["checkpoint_sha256"],
                }
            )
    except EvidenceIntegrityError as exc:
        checkpoint_error = str(exc)

    return {
        "default_mode": "checkpoint",
        "checkpoint_models": checkpoint_models,
        "checkpoint_available": bool(checkpoint_models)
        and config.CORE_PYTHON.is_file()
        and config.CHECKPOINT_WORKER.is_file(),
        "checkpoint_error": checkpoint_error,
        "min_date": str(min_date),
        "max_date": str(max_date),
        "default_date": str(max_date),
        "last_close": last_close,
        "last_date": str(df["date"].iloc[-1]),
        "model_train_horizon": logic.data.MODEL_TRAIN_HORIZON,
        "offline_available": bool(offline_dates) or config.OFFLINE_PREDICTION_PATH.exists(),
        # Offline backup is date-selectable over the held-out test split.
        # The full list lets the UI render a closed dropdown (no free-text dates).
        "offline_dates": offline_dates,
        "offline_min_date": offline_dates[0] if offline_dates else None,
        "offline_max_date": offline_dates[-1] if offline_dates else None,
        "offline_default_date": offline_dates[-1] if offline_dates else None,
        "api_url_default": config.API_URL_ENV,
    }


def run_checkpoint(req: CheckpointPredictionRequest) -> dict:
    """Run a hash-verified local checkpoint over the submitted causal window."""
    try:
        models = _checkpoint_models_from_evidence()
        expected = _verified_checkpoint_model(models, req.model_id)
    except EvidenceIntegrityError as exc:
        raise EvidenceUnavailableError(f"Verified checkpoint evidence is unavailable: {exc}") from exc

    window_days = expected["window_days"]
    window_rows = _checkpoint_window_rows(req, window_days=window_days)
    payload = req.model_dump(mode="json")
    try:
        prediction = run_checkpoint_inference(payload)
    except CheckpointInferenceError as exc:
        raise ConsoleError(exc.detail, status_code=exc.status_code) from exc

    provenance_fields = ("checkpoint_sha256", "source_commit", "window_days")
    mismatched = [
        field
        for field in provenance_fields
        if prediction.get(field) != expected[field]
    ]
    if mismatched:
        raise ConsoleError((
                "Checkpoint worker provenance does not match verified evidence: "
                f"{', '.join(mismatched)}."
            ),
        )
    if (
        prediction.get("window_start") != window_rows[0]["date"]
        or prediction.get("window_end") != window_rows[-1]["date"]
    ):
        raise ConsoleError("Checkpoint worker window metadata does not reconcile with the request.")
    window = pd.DataFrame(window_rows)
    worker_prediction = dict(prediction)
    worker_prediction["inference_type"] = "local_checkpoint"
    result = _result_from_prediction(
        worker_prediction,
        window,
        ood=bool(logic.data.is_out_of_distribution(window)),
        provenance=None,
    )
    result.update(
        {
            "inference_type": "local_checkpoint",
            "window_days": window_days,
            "expected_return_pct": float(prediction["expected_return_pct"]),
            "checkpoint_sha256": prediction["checkpoint_sha256"],
            "source_commit": prediction["source_commit"],
        }
    )
    return result


def run_live(req: LivePredictRequest) -> dict:
    """Run a real prediction via the remote Colab/ngrok model API."""
    df = _paper_df()
    try:
        window = logic.data.select_window(df, req.prediction_date) if req.prediction_date else df.tail(
            logic.data.MODEL_WINDOW_SIZE
        ).reset_index(drop=True)
    except logic.CandleDataError as exc:
        raise InvalidInputError(str(exc)) from exc

    payload = logic.data.build_predict_payload(df, prediction_date=req.prediction_date, risk=req.risk)
    ood = bool(logic.data.is_out_of_distribution(window))

    client = logic.CryptoMambaApiClient(base_url=req.api_url.rstrip("/"))
    try:
        prediction = client.predict(payload)
    except logic.ApiClientError as exc:
        raise UpstreamError(f"Model API error: {exc}") from exc

    if "predicted_close" not in prediction:
        raise UpstreamError("Model API response is missing 'predicted_close'.")
    prediction.setdefault("inference_type", "live")
    prediction.setdefault("last_close", float(window["close"].iloc[-1]))
    prediction.setdefault("prediction_date", payload["prediction_date"])
    return _result_from_prediction(prediction, window, ood=ood, provenance=None)


def run_offline(date: Optional[str] = None) -> dict:
    """Offline backup — date-selectable over the held-out test split using frozen
    Phase-2 predictions (real model outputs, no Colab). Falls back to the single
    golden fixture if forecast_predictions.csv is unavailable. Honest NOT_READY if
    nothing is present. Always labelled inference_type=offline (never live)."""
    ft = _forecast_test_df()
    if ft is not None:
        target = date or str(ft["prediction_date"].iloc[-1])  # default: latest test date
        res = _offline_by_date(target)
        if res is not None:
            return res

    # Fallback: the single frozen golden-fixture backup.
    try:
        bundle = logic.load_offline_prediction(config.OFFLINE_PREDICTION_PATH)
    except logic.OfflinePredictionError as exc:
        return {"available": False, "error": str(exc)}

    window = logic.data.window_from_candles(bundle["window_candles"])
    prediction = dict(bundle["prediction"])
    prediction["inference_type"] = "offline"  # hard rule: never present frozen as live
    ood = bool(logic.data.is_out_of_distribution(window))
    result = _result_from_prediction(prediction, window, ood=ood, provenance=bundle.get("provenance"))
    result["available"] = True
    return result
