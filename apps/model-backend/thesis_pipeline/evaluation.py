"""Date-aligned forecast evaluation over the common target dates.

Only models with per-date local predictions enter this module. Aggregate numbers
transcribed from the paper have no per-date series, so they are handled by the
artifact builder and can never reach a paired test.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from thesis_pipeline.contracts import MODEL_SPECS


DIR_EPS_REL = 1e-6
DM_HAC_LAG = 1
ALPHA = 0.05
SPLITS = ("val", "test")
MODEL_ORDER = (
    "cmamba_v_reproduced",
    "s5_full",
    "naive_persistence",
)
PAIRS = (
    ("cmamba_v_reproduced", "naive_persistence"),
    ("s5_full", "naive_persistence"),
    ("s5_full", "cmamba_v_reproduced"),
)

CANONICAL_COLUMNS = [
    "model_id",
    "display_name",
    "split",
    "signal_date",
    "prediction_date",
    "prediction_timestamp",
    "current_close",
    "target_close",
    "predicted_close",
    "parameter_count",
    "checkpoint_path",
    "checkpoint_sha256",
    "source_commit",
    "source_artifact",
]


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required thesis artifact is missing: {path}")
    return path


def _load_reproduced_predictions(core_root: Path, split: str) -> pd.DataFrame:
    path = _require_file(core_root / "output/evaluation/forecast_predictions.csv")
    frame = pd.read_csv(path)
    required = {
        "result_type",
        "split",
        "window_end_date",
        "prediction_date",
        "prediction_timestamp",
        "current_close",
        "target_close",
        "predicted_close",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"reproduced prediction columns missing: {sorted(missing)}")
    selected = frame[
        (frame["result_type"] == "retrained_checkpoint")
        & (frame["split"] == split)
    ].copy()
    if selected.empty:
        raise ValueError(f"no reproduced CM-v rows for split {split!r}")
    if selected["prediction_timestamp"].duplicated().any():
        raise ValueError(f"duplicate reproduced timestamps in split {split!r}")
    return selected.sort_values("prediction_timestamp").reset_index(drop=True)


def _load_s5_predictions(core_root: Path, split: str) -> tuple[pd.DataFrame, Path]:
    path = _require_file(
        core_root
        / "output/improve_track_evidence/s5_full/preds"
        / f"s5_full__seed23__{split}.csv"
    )
    frame = pd.read_csv(path)
    required = {"timestamp", "y", "y_hat", "y_old"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"S5-Full prediction columns missing: {sorted(missing)}")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"duplicate S5-Full timestamps in split {split!r}")
    frame["timestamp"] = frame["timestamp"].astype("int64")
    return frame.sort_values("timestamp").reset_index(drop=True), path


def _canonical_model_rows(
    aligned: pd.DataFrame,
    *,
    split: str,
    model_id: str,
    predicted_column: str,
    source_artifact: str,
) -> pd.DataFrame:
    spec = MODEL_SPECS[model_id]
    rows = pd.DataFrame(
        {
            "model_id": model_id,
            "display_name": spec.display_name,
            "split": split,
            "signal_date": aligned["window_end_date"].astype(str),
            "prediction_date": aligned["prediction_date"].astype(str),
            "prediction_timestamp": aligned["prediction_timestamp"].astype("int64"),
            "current_close": aligned["current_close"].astype(float),
            "target_close": aligned["target_close"].astype(float),
            "predicted_close": aligned[predicted_column].astype(float),
            "parameter_count": spec.parameter_count,
            "checkpoint_path": (
                str(spec.checkpoint_path) if spec.checkpoint_path is not None else None
            ),
            "checkpoint_sha256": spec.checkpoint_sha256,
            "source_commit": spec.source_commit,
            "source_artifact": source_artifact,
        }
    )
    return rows[CANONICAL_COLUMNS]


def build_aligned_predictions(core_root: Path) -> pd.DataFrame:
    """Build the three-model table on S5-Full's 304 common dates per split."""
    core_root = Path(core_root).resolve()
    frames: list[pd.DataFrame] = []
    cm_source = "output/evaluation/forecast_predictions.csv"
    for split in SPLITS:
        cm = _load_reproduced_predictions(core_root, split)
        s5, s5_path = _load_s5_predictions(core_root, split)
        aligned = cm.merge(
            s5,
            left_on="prediction_timestamp",
            right_on="timestamp",
            how="inner",
            validate="one_to_one",
        )
        if len(aligned) != 304:
            raise ValueError(
                f"split {split!r} must have 304 aligned rows, found {len(aligned)}"
            )
        converted_dates = pd.to_datetime(
            aligned["prediction_timestamp"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%d")
        if not converted_dates.equals(aligned["prediction_date"].astype(str)):
            raise ValueError(f"timestamp/date mismatch in split {split!r}")
        if not np.allclose(
            aligned["y"].to_numpy(float),
            aligned["target_close"].to_numpy(float),
            rtol=0.0,
            atol=1e-3,
        ):
            raise ValueError(f"S5-Full target prices disagree with CM-v in {split!r}")
        if not np.allclose(
            aligned["y_old"].to_numpy(float),
            aligned["current_close"].to_numpy(float),
            rtol=0.0,
            atol=1e-3,
        ):
            raise ValueError(f"S5-Full current prices disagree with CM-v in {split!r}")

        frames.append(
            _canonical_model_rows(
                aligned,
                split=split,
                model_id="cmamba_v_reproduced",
                predicted_column="predicted_close",
                source_artifact=cm_source,
            )
        )
        frames.append(
            _canonical_model_rows(
                aligned,
                split=split,
                model_id="s5_full",
                predicted_column="y_hat",
                source_artifact=str(s5_path.relative_to(core_root)),
            )
        )
        aligned["persistence_prediction"] = aligned["current_close"]
        frames.append(
            _canonical_model_rows(
                aligned,
                split=split,
                model_id="naive_persistence",
                predicted_column="persistence_prediction",
                source_artifact="derived: predicted_close=current_close",
            )
        )

    output = pd.concat(frames, ignore_index=True)
    split_order = output["split"].map({"val": 0, "test": 1})
    model_order = output["model_id"].map(
        {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    )
    output = (
        output.assign(_split_order=split_order, _model_order=model_order)
        .sort_values(["_split_order", "prediction_timestamp", "_model_order"])
        .drop(columns=["_split_order", "_model_order"])
        .reset_index(drop=True)
    )
    if output.duplicated(["model_id", "split", "prediction_date"]).any():
        raise ValueError("canonical predictions contain duplicate model/date keys")
    return output[CANONICAL_COLUMNS]


def _direction(delta: np.ndarray, current: np.ndarray) -> np.ndarray:
    epsilon = DIR_EPS_REL * np.abs(current)
    direction = np.zeros(delta.shape, dtype=np.int8)
    direction[delta > epsilon] = 1
    direction[delta < -epsilon] = -1
    return direction


def _validate_canonical(predictions: pd.DataFrame) -> None:
    required = set(CANONICAL_COLUMNS)
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"canonical prediction columns missing: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("canonical predictions must not be empty")
    if predictions.duplicated(["model_id", "split", "prediction_date"]).any():
        raise ValueError("canonical model/date keys must be unique")


def _group_metrics(group: pd.DataFrame) -> dict[str, Any]:
    model_id = str(group["model_id"].iloc[0])
    spec = MODEL_SPECS[model_id]
    current = group["current_close"].to_numpy(float)
    target = group["target_close"].to_numpy(float)
    predicted = group["predicted_close"].to_numpy(float)
    error = predicted - target
    actual_direction = _direction(target - current, current)
    predicted_direction = _direction(predicted - current, current)
    covered = predicted_direction != 0
    coverage = float(covered.mean() * 100.0)
    directional_accuracy = (
        float((predicted_direction[covered] == actual_direction[covered]).mean() * 100.0)
        if covered.any()
        else float("nan")
    )
    return {
        "model_id": model_id,
        "display_name": spec.display_name,
        "split": str(group["split"].iloc[0]),
        "samples": int(len(group)),
        "date_from": str(group["prediction_date"].min()),
        "date_to": str(group["prediction_date"].max()),
        "MSE": float(np.mean(error**2)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE_pct": float(np.mean(np.abs(error) / target) * 100.0),
        "directional_accuracy_pct": directional_accuracy,
        "directional_coverage_pct": coverage,
        "predicted_up_pct": float((predicted_direction > 0).mean() * 100.0),
        "parameter_count": spec.parameter_count,
        "checkpoint_sha256": spec.checkpoint_sha256,
    }


def forecast_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute split-specific metrics without pooling different date sets."""
    _validate_canonical(predictions)
    rows = [
        _group_metrics(group.sort_values("prediction_date"))
        for _, group in predictions.groupby(["split", "model_id"], sort=False)
    ]
    output = pd.DataFrame(rows)
    split_order = output["split"].map({"val": 0, "test": 1})
    model_order = output["model_id"].map(
        {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    )
    return (
        output.assign(_split_order=split_order, _model_order=model_order)
        .sort_values(["_split_order", "_model_order"])
        .drop(columns=["_split_order", "_model_order"])
        .reset_index(drop=True)
    )


def _diebold_mariano(
    error_a: np.ndarray, error_b: np.ndarray
) -> tuple[float, float, float]:
    differential = error_a**2 - error_b**2
    mean_differential = float(differential.mean())
    centered = differential - mean_differential
    gamma_0 = float(np.mean(centered**2))
    gamma_1 = float(np.mean(centered[1:] * centered[:-1]))
    long_run_variance = gamma_0 + 2.0 * gamma_1
    variance_of_mean = long_run_variance / len(differential)
    if variance_of_mean <= 0.0 or not math.isfinite(variance_of_mean):
        return float("nan"), float("nan"), mean_differential
    statistic = mean_differential / math.sqrt(variance_of_mean)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(statistic)))
    return float(statistic), float(p_value), mean_differential


def _aligned_pair(
    predictions: pd.DataFrame,
    *,
    split: str,
    model_a: str,
    model_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = predictions[
        (predictions["split"] == split) & (predictions["model_id"] == model_a)
    ].sort_values("prediction_date")
    right = predictions[
        (predictions["split"] == split) & (predictions["model_id"] == model_b)
    ].sort_values("prediction_date")
    if len(left) == 0 or len(left) != len(right):
        raise ValueError(f"unaligned pair {model_a!r} vs {model_b!r} on {split!r}")
    if left["prediction_date"].tolist() != right["prediction_date"].tolist():
        raise ValueError(f"date mismatch for {model_a!r} vs {model_b!r}")
    for column in ("current_close", "target_close"):
        if not np.allclose(
            left[column].to_numpy(float),
            right[column].to_numpy(float),
            rtol=0.0,
            atol=1e-3,
        ):
            raise ValueError(f"actual-price mismatch for paired column {column!r}")
    return left, right


def paired_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    """Run paired tests only on locally available, identical target dates."""
    _validate_canonical(predictions)
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for model_a, model_b in PAIRS:
            left, right = _aligned_pair(
                predictions, split=split, model_a=model_a, model_b=model_b
            )
            target = left["target_close"].to_numpy(float)
            error_a = left["predicted_close"].to_numpy(float) - target
            error_b = right["predicted_close"].to_numpy(float) - target
            dm_statistic, dm_p_value, mean_squared_differential = _diebold_mariano(
                error_a, error_b
            )
            absolute_differential = np.abs(error_a) - np.abs(error_b)
            if np.allclose(absolute_differential, 0.0):
                wilcoxon_statistic = float("nan")
                wilcoxon_p_value = float("nan")
            else:
                wilcoxon = stats.wilcoxon(
                    absolute_differential,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                wilcoxon_statistic = float(wilcoxon.statistic)
                wilcoxon_p_value = float(wilcoxon.pvalue)
            mse_a = float(np.mean(error_a**2))
            mse_b = float(np.mean(error_b**2))
            mae_a = float(np.mean(np.abs(error_a)))
            mae_b = float(np.mean(np.abs(error_b)))
            rows.append(
                {
                    "split": split,
                    "model_a": model_a,
                    "model_b": model_b,
                    "comparison": f"{model_a} vs {model_b}",
                    "samples": int(len(left)),
                    "date_from": str(left["prediction_date"].iloc[0]),
                    "date_to": str(left["prediction_date"].iloc[-1]),
                    "dm_loss": "squared_error",
                    "dm_hac_lag": DM_HAC_LAG,
                    "dm_statistic": dm_statistic,
                    "dm_p_value": dm_p_value,
                    "mean_squared_loss_differential_a_minus_b": mean_squared_differential,
                    "wilcoxon_loss": "absolute_error",
                    "wilcoxon_statistic": wilcoxon_statistic,
                    "wilcoxon_p_value": wilcoxon_p_value,
                    "mean_absolute_loss_differential_a_minus_b": mae_a - mae_b,
                    "lower_mse_model": model_a if mse_a < mse_b else model_b,
                    "lower_mae_model": model_a if mae_a < mae_b else model_b,
                    "dm_significant_5pct": bool(
                        math.isfinite(dm_p_value) and dm_p_value < ALPHA
                    ),
                    "wilcoxon_significant_5pct": bool(
                        math.isfinite(wilcoxon_p_value)
                        and wilcoxon_p_value < ALPHA
                    ),
                }
            )
    return pd.DataFrame(rows)
