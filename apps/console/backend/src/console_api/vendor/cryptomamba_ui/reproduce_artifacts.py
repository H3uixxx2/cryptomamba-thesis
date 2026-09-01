from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


FORECAST_COLUMNS = {
    "result_type",
    "model",
    "split",
    "checkpoint",
    "source_commit",
    "protocol",
    "samples",
    "RMSE",
    "MAE",
    "MAPE_pct",
    "paper_RMSE",
    "paper_MAE",
    "paper_MAPE_pct",
    "RMSE_gap_pct",
    "MAE_gap_pct",
    "MAPE_gap_pct",
    "tolerance_pct",
    "status",
}
REPLAY_COLUMNS = {
    "result_type",
    "model",
    "checkpoint",
    "source_commit",
    "split",
    "trade_mode",
    "initial_balance",
    "risk_pct",
    "final_balance",
    "max_drawdown_pct",
    "paper_final_balance",
    "paper_max_drawdown_pct",
    "balance_gap_pct",
    "mdd_gap_pp",
    "protocol",
    "status",
}
BASELINE_COLUMNS = {"model", "split", "samples", "RMSE", "MAE", "MAPE_pct", "protocol"}
# The mandatory baseline set. COMPLETE only when all four are present.
MANDATORY_BASELINES = {"naive_persistence", "lstm", "gru", "itransformer"}
PREDICTION_COLUMNS = {
    "result_type",
    "model",
    "checkpoint",
    "source_commit",
    "split",
    "sample_index",
    "window_start_date",
    "window_end_date",
    "prediction_date",
    "current_close",
    "target_close",
    "predicted_close",
    "actual_return_pct",
    "predicted_return_pct",
    "protocol",
}
EXPECTED_RESULT_TYPES = {"official_checkpoint", "retrained_checkpoint"}
EXPECTED_REPLAY_KEYS = {
    (result_type, split, mode)
    for result_type in EXPECTED_RESULT_TYPES
    for split in ("val", "test")
    for mode in ("vanilla", "smart", "smart_w_short")
}


@dataclass
class ReproduceArtifacts:
    status: str = "NOT_READY"
    forecast_status: str = "NOT_READY"
    replay_status: str = "NOT_READY"
    baseline_status: str = "NOT_READY"
    forecast_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    trading_replay_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    significance: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_models_present: tuple[str, ...] = ()
    forecast_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_selection: dict[str, Any] = field(default_factory=dict)
    inference_fixture: dict[str, Any] = field(default_factory=dict)
    artifact_validation: dict[str, Any] = field(default_factory=dict)
    source_contract: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()


def load_reproduce_artifacts(
    evaluation_dir: Path,
    provenance_dir: Path,
    selected_checkpoint_path: Path,
) -> ReproduceArtifacts:
    errors: list[str] = []
    forecast = _read_csv(evaluation_dir / "forecast_metrics.csv", errors)
    replay = _read_csv(evaluation_dir / "trading_replay_metrics.csv", errors)
    baseline = _read_csv(evaluation_dir / "baseline_metrics.csv", errors)
    baseline_comparison = _read_csv_optional(evaluation_dir / "baseline_metrics_comparison.csv")
    significance = _read_csv_optional(evaluation_dir / "significance_tests.csv")
    predictions = _read_csv(evaluation_dir / "forecast_predictions.csv", errors)
    baseline_predictions = _read_csv_optional(evaluation_dir / "baseline_predictions.csv")
    model_selection = _read_json(evaluation_dir / "model_selection.json", errors)
    inference_fixture = _read_json(evaluation_dir / "inference_fixture.json", errors)
    artifact_validation = _read_json(provenance_dir / "artifact_validation.json", errors)
    source_contract = _read_json(provenance_dir / "source_contract.json", errors)
    evaluation_parity = _read_json(provenance_dir / "source_cli_parity/evaluation_parity.json", errors)
    trading_parity = _read_json(provenance_dir / "source_cli_parity/trading_parity.json", errors)

    forecast_status = _validate_forecast(forecast, errors)
    replay_status = _validate_replay(replay, errors)
    baseline_status, baseline_models_present = _validate_baseline(baseline, baseline_comparison, errors)
    _validate_predictions(predictions, errors)
    selected_hash = _validate_selection(
        model_selection,
        inference_fixture,
        artifact_validation,
        selected_checkpoint_path,
        errors,
    )

    if artifact_validation and artifact_validation.get("status") != "PASS":
        errors.append(f"artifact_validation status is {artifact_validation.get('status')!r}, expected 'PASS'")
    if artifact_validation:
        expected_counts = {
            "forecast_rows": len(forecast),
            "prediction_rows": len(predictions),
            "replay_rows": len(replay),
            "baseline_rows": len(baseline),
        }
        for field_name, actual_count in expected_counts.items():
            if artifact_validation.get(field_name) != actual_count:
                errors.append(
                    f"artifact_validation {field_name}={artifact_validation.get(field_name)!r} "
                    f"does not match loaded rows={actual_count}"
                )
    if evaluation_parity and evaluation_parity.get("status") != "PASS":
        errors.append("released evaluation CLI parity is not PASS")
    if trading_parity and trading_parity.get("status") != "PASS":
        errors.append("released trading CLI parity is not PASS")
    if source_contract and source_contract.get("config") != "cmamba_v":
        errors.append("source_contract config is not cmamba_v")
    if artifact_validation and selected_hash:
        validation_hash = artifact_validation.get("selected_checkpoint_sha256")
        if validation_hash != selected_hash:
            errors.append("artifact_validation selected checkpoint hash does not match model_selection")

    ready = (
        not errors
        and forecast_status == "PASS"
        and replay_status in {"VERIFIED", "COMPLETE_WITH_RETRAINED_MISMATCH"}
        and baseline_status in {"COMPLETE", "PARTIAL"}
    )
    return ReproduceArtifacts(
        status="READY" if ready else "NOT_READY",
        forecast_status=forecast_status,
        replay_status=replay_status,
        baseline_status=baseline_status,
        forecast_metrics=forecast,
        trading_replay_metrics=replay,
        baseline_metrics=baseline,
        baseline_comparison=baseline_comparison,
        significance=significance,
        baseline_models_present=baseline_models_present,
        forecast_predictions=predictions,
        baseline_predictions=baseline_predictions,
        model_selection=model_selection,
        inference_fixture=inference_fixture,
        artifact_validation=artifact_validation,
        source_contract=source_contract,
        errors=tuple(errors),
    )


def _read_csv(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.is_file():
        errors.append(f"Missing required artifact: {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        errors.append(f"Cannot parse {path.name}: {exc}")
        return pd.DataFrame()


def _read_csv_optional(path: Path) -> pd.DataFrame:
    """Read a supplementary CSV. Missing/unreadable file is not an error (returns empty)."""
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"Missing required artifact: {path.name}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot parse {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return {}
    return payload


def _validate_columns(frame: pd.DataFrame, required: set[str], artifact_name: str, errors: list[str]) -> bool:
    if frame.empty:
        return False
    missing = sorted(required.difference(frame.columns))
    if missing:
        errors.append(f"{artifact_name} missing required columns: {', '.join(missing)}")
        return False
    return True


def _validate_forecast(frame: pd.DataFrame, errors: list[str]) -> str:
    if not _validate_columns(frame, FORECAST_COLUMNS, "forecast_metrics.csv", errors):
        return "NOT_READY"
    if frame.duplicated(["result_type", "model", "split"]).any():
        errors.append("forecast_metrics.csv contains duplicate logical rows")
        return "NOT_READY"
    if set(frame["result_type"]) != EXPECTED_RESULT_TYPES or set(frame["split"]) != {"test"}:
        errors.append("forecast_metrics.csv must contain separate official and retrained test rows")
        return "NOT_READY"

    numeric_columns = [
        "samples",
        "RMSE",
        "MAE",
        "MAPE_pct",
        "paper_RMSE",
        "paper_MAE",
        "paper_MAPE_pct",
        "RMSE_gap_pct",
        "MAE_gap_pct",
        "MAPE_gap_pct",
        "tolerance_pct",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.apply(lambda column: column.map(math.isfinite)).all().all():
        errors.append("forecast_metrics.csv contains non-finite numeric values")
        return "NOT_READY"
    if (numeric < 0).any().any():
        errors.append("forecast_metrics.csv contains negative metrics")
        return "NOT_READY"

    error_count_before_gap_check = len(errors)
    metric_columns = (
        ("RMSE", "paper_RMSE", "RMSE_gap_pct"),
        ("MAE", "paper_MAE", "MAE_gap_pct"),
        ("MAPE_pct", "paper_MAPE_pct", "MAPE_gap_pct"),
    )
    for index, row in frame.iterrows():
        for metric, paper_column, gap_column in metric_columns:
            expected_gap = abs(float(row[metric]) - float(row[paper_column])) / float(row[paper_column]) * 100
            if not math.isclose(float(row[gap_column]), expected_gap, rel_tol=1e-5, abs_tol=1e-3):
                errors.append(f"forecast_metrics.csv {gap_column} is inconsistent for row {index}")
    if len(errors) > error_count_before_gap_check:
        return "NOT_READY"

    gap_columns = ["RMSE_gap_pct", "MAE_gap_pct", "MAPE_gap_pct"]
    gaps_pass = numeric[gap_columns].le(numeric["tolerance_pct"], axis=0).all().all()
    statuses_pass = frame["status"].astype(str).str.upper().eq("PASS").all()
    if not gaps_pass or not statuses_pass:
        return "FAIL"
    return "PASS"


def _validate_replay(frame: pd.DataFrame, errors: list[str]) -> str:
    if not _validate_columns(frame, REPLAY_COLUMNS, "trading_replay_metrics.csv", errors):
        return "NOT_READY"
    numeric_columns = [
        "initial_balance",
        "risk_pct",
        "final_balance",
        "max_drawdown_pct",
        "paper_final_balance",
        "paper_max_drawdown_pct",
        "balance_gap_pct",
        "mdd_gap_pp",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.apply(lambda column: column.map(math.isfinite)).all().all():
        errors.append("trading_replay_metrics.csv contains non-finite numeric values")
        return "NOT_READY"
    non_negative_columns = numeric_columns[:6]
    if (numeric[non_negative_columns] < 0).any().any():
        errors.append("trading_replay_metrics.csv contains negative balance/risk/drawdown values")
        return "NOT_READY"
    if frame.duplicated(["result_type", "model", "split", "trade_mode"]).any():
        errors.append("trading_replay_metrics.csv contains duplicate logical rows")
        return "NOT_READY"
    actual_keys = set(frame[["result_type", "split", "trade_mode"]].itertuples(index=False, name=None))
    if actual_keys != EXPECTED_REPLAY_KEYS:
        errors.append("trading_replay_metrics.csv does not contain all 12 official/retrained replay rows")
        return "INCOMPLETE"

    official_status = frame.loc[frame["result_type"] == "official_checkpoint", "status"].astype(str).str.upper()
    if not official_status.eq("VERIFIED").all():
        errors.append("official checkpoint replay rows are not all VERIFIED")
        return "NOT_READY"
    retrained_status = frame.loc[frame["result_type"] == "retrained_checkpoint", "status"].astype(str).str.upper()
    return "COMPLETE_WITH_RETRAINED_MISMATCH" if retrained_status.eq("NOT_MATCHED").any() else "VERIFIED"


def _validate_baseline(
    frame: pd.DataFrame, comparison: pd.DataFrame, errors: list[str]
) -> tuple[str, tuple[str, ...]]:
    """Validate baseline_metrics.csv and derive COMPLETE/PARTIAL against the full mandatory set.

    The mandatory baselines are {naive, lstm, gru, itransformer}. Models are collected from
    baseline_metrics.csv (test split) AND, when present, the unified baseline_metrics_comparison.csv
    (which is where the neural baselines land). COMPLETE only when all four are present;
    otherwise PARTIAL. Returns (status, present_mandatory_models).
    """
    if not _validate_columns(frame, BASELINE_COLUMNS, "baseline_metrics.csv", errors):
        return "NOT_READY", ()
    numeric_columns = ["samples", "RMSE", "MAE", "MAPE_pct"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.apply(lambda column: column.map(math.isfinite)).all().all():
        errors.append("baseline_metrics.csv contains non-finite numeric values")
        return "NOT_READY", ()
    if (numeric < 0).any().any():
        errors.append("baseline_metrics.csv contains negative metrics")
        return "NOT_READY", ()
    test_rows = frame[frame["split"] == "test"]
    if test_rows.empty:
        errors.append("baseline_metrics.csv has no test split")
        return "NOT_READY", ()
    models = set(test_rows["model"].astype(str))
    if not comparison.empty and "model" in comparison.columns:
        models |= set(comparison["model"].astype(str))
    present = tuple(sorted(MANDATORY_BASELINES & models))
    status = "COMPLETE" if MANDATORY_BASELINES.issubset(models) else "PARTIAL"
    return status, present


def _validate_predictions(frame: pd.DataFrame, errors: list[str]) -> None:
    if not _validate_columns(frame, PREDICTION_COLUMNS, "forecast_predictions.csv", errors):
        return
    if frame.duplicated(["result_type", "split", "sample_index"]).any():
        errors.append("forecast_predictions.csv contains duplicate logical rows")
    if not EXPECTED_RESULT_TYPES.issubset(set(frame["result_type"])):
        errors.append("forecast_predictions.csv must include official and retrained predictions")
    for column in ("window_start_date", "window_end_date", "prediction_date"):
        if pd.to_datetime(frame[column], errors="coerce").isna().any():
            errors.append(f"forecast_predictions.csv contains invalid {column}")
    if frame["source_commit"].astype(str).str.strip().eq("").any():
        errors.append("forecast_predictions.csv contains empty source_commit")


def _validate_selection(
    model_selection: dict[str, Any],
    inference_fixture: dict[str, Any],
    artifact_validation: dict[str, Any],
    checkpoint_path: Path,
    errors: list[str],
) -> str:
    selection_fields = {
        "selected_checkpoint_type",
        "selected_checkpoint_path",
        "checkpoint_sha256",
        "source_commit",
        "selection_reason",
        "forecast_status",
        "fallback_checkpoint",
    }
    missing_selection = sorted(selection_fields.difference(model_selection))
    if missing_selection:
        errors.append(f"model_selection.json missing fields: {', '.join(missing_selection)}")
        return ""
    if model_selection.get("forecast_status") != "PASS":
        errors.append("model_selection forecast_status is not PASS")
    if model_selection.get("selected_checkpoint_type") != "retrained_checkpoint":
        errors.append("model_selection did not select the validated retrained checkpoint")

    selected_hash = str(model_selection.get("checkpoint_sha256", ""))
    fixture_hash = inference_fixture.get("checkpoint_sha256")
    if fixture_hash != selected_hash:
        errors.append("inference fixture checkpoint hash does not match model_selection")
    if len(inference_fixture.get("candles", [])) != 14:
        errors.append("inference_fixture.json must contain 14 candles")
    if inference_fixture.get("expected_feature_order") != ["Timestamp", "Open", "High", "Low", "Close", "Volume"]:
        errors.append("inference_fixture.json feature order is invalid")
    if inference_fixture.get("expected_tensor_shape") != [1, 6, 14]:
        errors.append("inference_fixture.json tensor shape is invalid")

    validation_hash = artifact_validation.get("selected_checkpoint_sha256")
    if validation_hash and validation_hash != selected_hash:
        errors.append("artifact_validation selected checkpoint hash does not match model_selection")
    if not checkpoint_path.is_file():
        errors.append(f"Selected checkpoint is missing: {checkpoint_path}")
    elif _sha256(checkpoint_path) != selected_hash:
        errors.append("Selected checkpoint file hash does not match model_selection")
    return selected_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
