"""Deterministic robustness analysis over checksum-verified daily predictions."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from console_api.loaders.final_evidence import EvidenceIntegrityError, FinalEvidence


MODEL_ORDER = ("cmamba_v_reproduced", "s5_full", "naive_persistence")
SPLITS = (("val", "Val"), ("test", "Test"))
SOURCE_ARTIFACT = "forecast/controlled_predictions.csv"
SOURCE_MANIFEST = "provenance/source_artifact_manifest.json"
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_RANDOM_SEED = 230_813
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
PRIMARY_BLOCK_LENGTH = 7
BLOCK_LENGTHS = (5, 7, 14)
EXPECTED_SAMPLES = 304
_BATCH_SIZE = 2_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _protocol() -> dict[str, Any]:
    return {
        "method": "paired non-circular moving-block bootstrap",
        "resamples": BOOTSTRAP_RESAMPLES,
        "primary_block_length": PRIMARY_BLOCK_LENGTH,
        "block_lengths": list(BLOCK_LENGTHS),
        "random_seed": BOOTSTRAP_RANDOM_SEED,
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "retraining": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(evidence: FinalEvidence) -> dict[str, Any]:
    manifest = evidence.read_json(SOURCE_MANIFEST)
    try:
        entry = manifest["files"]["controlled_predictions.csv"]
        expected_sha256 = entry["sha256"]
        expected_bytes = entry["bytes"]
    except (KeyError, TypeError) as exc:
        raise EvidenceIntegrityError(
            "source manifest does not describe controlled_predictions.csv"
        ) from exc
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise EvidenceIntegrityError("invalid controlled-prediction checksum metadata")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise EvidenceIntegrityError("invalid controlled-prediction byte metadata")

    path = evidence.root / "forecast" / "controlled_predictions.csv"
    try:
        actual_sha256 = _sha256(path)
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise EvidenceIntegrityError("controlled predictions are unreadable") from exc
    if actual_sha256 != expected_sha256 or actual_bytes != expected_bytes:
        raise EvidenceIntegrityError(
            "controlled-prediction provenance does not match the verified artifact"
        )
    return {
        "source_artifact": SOURCE_ARTIFACT,
        "source_sha256": expected_sha256,
        "source_rows": len(evidence.read_csv(SOURCE_ARTIFACT)),
        "samples_per_model_split": EXPECTED_SAMPLES,
        "evidence_package_sha256": evidence.package_sha256,
    }


def _load_controlled_predictions(evidence: FinalEvidence) -> pd.DataFrame:
    predictions = evidence.read_csv(SOURCE_ARTIFACT)
    required = {
        "model_id",
        "split",
        "prediction_date",
        "current_close",
        "target_close",
        "predicted_close",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise EvidenceIntegrityError(
            f"controlled prediction columns missing: {sorted(missing)}"
        )
    if set(predictions["model_id"]) != set(MODEL_ORDER):
        raise EvidenceIntegrityError("controlled predictions have unexpected models")
    if set(predictions["split"]) != {split for split, _ in SPLITS}:
        raise EvidenceIntegrityError("controlled predictions have unexpected splits")
    if predictions.duplicated(["model_id", "split", "prediction_date"]).any():
        raise EvidenceIntegrityError("controlled predictions contain duplicate model/date rows")
    group_sizes = predictions.groupby(["model_id", "split"]).size()
    if len(group_sizes) != 6 or any(int(size) != EXPECTED_SAMPLES for size in group_sizes):
        raise EvidenceIntegrityError(
            f"controlled predictions require {EXPECTED_SAMPLES} rows per model/split"
        )

    predictions = predictions.copy()
    try:
        predictions["prediction_date"] = pd.to_datetime(
            predictions["prediction_date"], format="%Y-%m-%d", errors="raise"
        )
        for column in ("current_close", "target_close", "predicted_close"):
            predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvidenceIntegrityError("controlled prediction values are invalid") from exc
    if not np.isfinite(
        predictions[["current_close", "target_close", "predicted_close"]].to_numpy(
            dtype=float
        )
    ).all():
        raise EvidenceIntegrityError("controlled predictions contain non-finite prices")
    return predictions


def _aligned_prediction_arrays(
    predictions: pd.DataFrame, split: str
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    model_frames: dict[str, pd.DataFrame] = {}
    for model_id in MODEL_ORDER:
        rows = predictions[
            (predictions["split"] == split) & (predictions["model_id"] == model_id)
        ].sort_values("prediction_date")
        model_frames[model_id] = rows.reset_index(drop=True)

    reference = model_frames[MODEL_ORDER[0]]
    dates = reference["prediction_date"].to_numpy()
    current = reference["current_close"].to_numpy(dtype=float)
    target = reference["target_close"].to_numpy(dtype=float)
    predicted: dict[str, np.ndarray] = {}
    for model_id, rows in model_frames.items():
        if not np.array_equal(rows["prediction_date"].to_numpy(), dates):
            raise EvidenceIntegrityError(
                f"prediction dates are not aligned for {model_id}/{split}"
            )
        if not np.allclose(
            rows["current_close"].to_numpy(dtype=float), current, rtol=0.0, atol=1e-9
        ):
            raise EvidenceIntegrityError(
                f"current close differs for {model_id}/{split}"
            )
        if not np.allclose(
            rows["target_close"].to_numpy(dtype=float), target, rtol=0.0, atol=1e-9
        ):
            raise EvidenceIntegrityError(
                f"target close differs for {model_id}/{split}"
            )
        predicted[model_id] = rows["predicted_close"].to_numpy(dtype=float)
    return current, target, predicted, dates


def _exact_mcnemar_p(model_a_only_correct: int, model_b_only_correct: int) -> float:
    discordant = model_a_only_correct + model_b_only_correct
    if discordant == 0:
        return 1.0
    lower = min(model_a_only_correct, model_b_only_correct)
    lower_tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(values, (tail, 1.0 - tail))
    return float(low), float(high)


def _compute_block(predictions: pd.DataFrame, block_length: int) -> dict[str, list[dict]]:
    sample_count = EXPECTED_SAMPLES
    blocks_per_resample = math.ceil(sample_count / block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_RANDOM_SEED))

    rmse_rows: list[dict[str, Any]] = []
    rmse_difference_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    direction_difference_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []

    for split, split_label in SPLITS:
        current, target, predicted, dates = _aligned_prediction_arrays(predictions, split)
        errors = {model_id: predicted[model_id] - target for model_id in MODEL_ORDER}
        actual_up = target > current
        correct = {
            model_id: (predicted[model_id] > current) == actual_up
            for model_id in ("cmamba_v_reproduced", "s5_full")
        }
        rmse_bootstrap = {
            model_id: np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
            for model_id in MODEL_ORDER
        }
        direction_bootstrap = {
            model_id: np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
            for model_id in ("cmamba_v_reproduced", "s5_full")
        }
        for offset in range(0, BOOTSTRAP_RESAMPLES, _BATCH_SIZE):
            batch_size = min(_BATCH_SIZE, BOOTSTRAP_RESAMPLES - offset)
            starts = generator.integers(
                0,
                sample_count - block_length + 1,
                size=(batch_size, blocks_per_resample),
            )
            indices = (starts[..., None] + offsets).reshape(batch_size, -1)[
                :, :sample_count
            ]
            target_slice = slice(offset, offset + batch_size)
            for model_id in MODEL_ORDER:
                rmse_bootstrap[model_id][target_slice] = np.sqrt(
                    np.mean(np.square(errors[model_id][indices]), axis=1)
                )
            for model_id in ("cmamba_v_reproduced", "s5_full"):
                direction_bootstrap[model_id][target_slice] = (
                    np.mean(correct[model_id][indices], axis=1) * 100.0
                )

        for model_id in MODEL_ORDER:
            low, high = _percentile_interval(rmse_bootstrap[model_id])
            rmse_rows.append(
                {
                    "split": split,
                    "model_id": model_id,
                    "estimate": float(np.sqrt(np.mean(np.square(errors[model_id])))),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        rmse_delta = rmse_bootstrap["s5_full"] - rmse_bootstrap["cmamba_v_reproduced"]
        rmse_delta_low, rmse_delta_high = _percentile_interval(rmse_delta)
        rmse_difference_rows.append(
            {
                "split": split,
                "comparison": "s5_full-minus-cmamba_v_reproduced",
                "estimate": float(
                    np.sqrt(np.mean(np.square(errors["s5_full"])))
                    - np.sqrt(np.mean(np.square(errors["cmamba_v_reproduced"])))
                ),
                "ci_low": rmse_delta_low,
                "ci_high": rmse_delta_high,
            }
        )

        for model_id in ("cmamba_v_reproduced", "s5_full"):
            low, high = _percentile_interval(direction_bootstrap[model_id])
            direction_rows.append(
                {
                    "split": split,
                    "model_id": model_id,
                    "estimate_pct": float(np.mean(correct[model_id]) * 100.0),
                    "ci_low_pct": low,
                    "ci_high_pct": high,
                }
            )
        cm_only = int(
            np.sum(correct["cmamba_v_reproduced"] & ~correct["s5_full"])
        )
        s5_only = int(
            np.sum(~correct["cmamba_v_reproduced"] & correct["s5_full"])
        )
        direction_delta = (
            direction_bootstrap["cmamba_v_reproduced"]
            - direction_bootstrap["s5_full"]
        )
        direction_delta_low, direction_delta_high = _percentile_interval(direction_delta)
        direction_difference_rows.append(
            {
                "split": split,
                "comparison": "cmamba_v_reproduced-minus-s5_full",
                "estimate_pp": float(
                    (
                        np.mean(correct["cmamba_v_reproduced"])
                        - np.mean(correct["s5_full"])
                    )
                    * 100.0
                ),
                "ci_low_pp": direction_delta_low,
                "ci_high_pp": direction_delta_high,
                "model_a_only_correct": cm_only,
                "model_b_only_correct": s5_only,
                "mcnemar_exact_p": _exact_mcnemar_p(cm_only, s5_only),
            }
        )

        for half_index, positions in enumerate(np.array_split(np.arange(sample_count), 2), 1):
            period_errors = {
                model_id: errors[model_id][positions] for model_id in MODEL_ORDER
            }
            period_rows.append(
                {
                    "period": f"{split_label} H{half_index}",
                    "samples": int(len(positions)),
                    "date_from": pd.Timestamp(dates[positions[0]]).strftime("%Y-%m-%d"),
                    "date_to": pd.Timestamp(dates[positions[-1]]).strftime("%Y-%m-%d"),
                    "cmamba_v_rmse": float(
                        np.sqrt(
                            np.mean(np.square(period_errors["cmamba_v_reproduced"]))
                        )
                    ),
                    "s5_full_rmse": float(
                        np.sqrt(np.mean(np.square(period_errors["s5_full"])))
                    ),
                    "persistence_rmse": float(
                        np.sqrt(np.mean(np.square(period_errors["naive_persistence"])))
                    ),
                    "cmamba_v_direction_pct": float(
                        np.mean(correct["cmamba_v_reproduced"][positions]) * 100.0
                    ),
                    "s5_full_direction_pct": float(
                        np.mean(correct["s5_full"][positions]) * 100.0
                    ),
                }
            )
    return {
        "rmse": rmse_rows,
        "rmse_difference": rmse_difference_rows,
        "direction": direction_rows,
        "direction_difference": direction_difference_rows,
        "periods": period_rows,
    }


def _compute(evidence: FinalEvidence) -> dict[str, Any]:
    predictions = _load_controlled_predictions(evidence)
    metadata = _source_metadata(evidence)
    by_block = {
        block_length: _compute_block(predictions, block_length)
        for block_length in BLOCK_LENGTHS
    }
    primary = by_block[PRIMARY_BLOCK_LENGTH]

    sensitivity: list[dict[str, Any]] = []
    for block_length in BLOCK_LENGTHS:
        result = by_block[block_length]
        rmse_by_split = {row["split"]: row for row in result["rmse_difference"]}
        direction_by_split = {
            row["split"]: row for row in result["direction_difference"]
        }
        for split, _ in SPLITS:
            rmse = rmse_by_split[split]
            direction = direction_by_split[split]
            sensitivity.append(
                {
                    "split": split,
                    "block_length": block_length,
                    "rmse_difference_estimate": rmse["estimate"],
                    "rmse_ci_low": rmse["ci_low"],
                    "rmse_ci_high": rmse["ci_high"],
                    "rmse_ci_includes_zero": rmse["ci_low"] <= 0.0 <= rmse["ci_high"],
                    "direction_difference_estimate_pp": direction["estimate_pp"],
                    "direction_ci_low_pp": direction["ci_low_pp"],
                    "direction_ci_high_pp": direction["ci_high_pp"],
                    "direction_ci_includes_zero": (
                        direction["ci_low_pp"] <= 0.0 <= direction["ci_high_pp"]
                    ),
                }
            )
    return {
        "status": "READY",
        "error": None,
        "protocol": _protocol(),
        "rmse": primary["rmse"],
        "rmse_difference": primary["rmse_difference"],
        "direction": primary["direction"],
        "direction_difference": primary["direction_difference"],
        "periods": primary["periods"],
        "sensitivity": sensitivity,
        "evidence": metadata,
    }


@lru_cache(maxsize=4)
def _cached(root: str, package_sha256: str) -> dict[str, Any]:
    evidence = FinalEvidence(Path(root))
    if evidence.package_sha256 != package_sha256:
        raise EvidenceIntegrityError("evidence package changed before robustness computation")
    return _compute(evidence)


def get_forecast_robustness(evidence: FinalEvidence) -> dict[str, Any]:
    """Compute once per verified evidence root/package pair."""
    return _cached(str(evidence.root), evidence.package_sha256)


def not_ready_robustness(error: str) -> dict[str, Any]:
    return {
        "status": "NOT_READY",
        "error": error,
        "protocol": _protocol(),
        "rmse": [],
        "rmse_difference": [],
        "direction": [],
        "direction_difference": [],
        "periods": [],
        "sensitivity": [],
        "evidence": {},
    }


__all__ = ["get_forecast_robustness", "not_ready_robustness"]
