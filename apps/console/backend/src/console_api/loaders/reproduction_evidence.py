"""Recompute the 350-day CM-v reproduction from its pinned source artifact."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from console_api.loaders.final_evidence import EvidenceIntegrityError, FinalEvidence


SOURCE_ARTIFACT = "output/evaluation/forecast_predictions.csv"
SOURCE_MANIFEST = "provenance/source_artifact_manifest.json"
EXPECTED_RESULT_TYPES = ("official_checkpoint", "retrained_checkpoint")
EXPECTED_SAMPLES = 350
REPRODUCTION_THRESHOLD_PCT = 5.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _pinned_source(evidence: FinalEvidence, core_root: Path) -> tuple[bytes, str, int]:
    manifest = evidence.read_json(SOURCE_MANIFEST)
    try:
        entry = manifest["source_artifacts"][SOURCE_ARTIFACT]
        expected_sha256 = entry["sha256"]
        expected_bytes = entry["bytes"]
    except (KeyError, TypeError) as exc:
        raise EvidenceIntegrityError(
            f"source manifest does not pin {SOURCE_ARTIFACT}"
        ) from exc
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise EvidenceIntegrityError(f"invalid source checksum for {SOURCE_ARTIFACT}")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise EvidenceIntegrityError(f"invalid source byte count for {SOURCE_ARTIFACT}")

    root = Path(core_root).resolve()
    path = root.joinpath(*SOURCE_ARTIFACT.split("/"))
    if path.is_symlink() or not path.is_file():
        raise EvidenceIntegrityError(f"source artifact is missing: {SOURCE_ARTIFACT}")
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceIntegrityError(
            f"source artifact escapes configured core root: {SOURCE_ARTIFACT}"
        ) from exc
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise EvidenceIntegrityError(
            f"source artifact is unreadable: {SOURCE_ARTIFACT}"
        ) from exc
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    actual_bytes = len(source_bytes)
    if actual_sha256 != expected_sha256:
        raise EvidenceIntegrityError(f"source artifact checksum mismatch: {SOURCE_ARTIFACT}")
    if actual_bytes != expected_bytes:
        raise EvidenceIntegrityError(f"source artifact byte count mismatch: {SOURCE_ARTIFACT}")
    return source_bytes, expected_sha256, expected_bytes


def _load_test_predictions(source_bytes: bytes) -> dict[str, pd.DataFrame]:
    try:
        predictions = pd.read_csv(io.BytesIO(source_bytes))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise EvidenceIntegrityError(
            f"source artifact is unreadable: {SOURCE_ARTIFACT}"
        ) from exc

    required = {
        "result_type",
        "split",
        "prediction_date",
        "target_close",
        "predicted_close",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise EvidenceIntegrityError(
            f"source artifact columns missing: {sorted(missing)}"
        )
    selected = predictions[
        (predictions["split"] == "test")
        & predictions["result_type"].isin(EXPECTED_RESULT_TYPES)
    ].copy()
    try:
        selected["prediction_date"] = pd.to_datetime(
            selected["prediction_date"], format="%Y-%m-%d", errors="raise"
        )
        for column in ("target_close", "predicted_close"):
            selected[column] = pd.to_numeric(selected[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvidenceIntegrityError("350-day prediction values are invalid") from exc
    if not np.isfinite(
        selected[["target_close", "predicted_close"]].to_numpy(dtype=float)
    ).all():
        raise EvidenceIntegrityError("350-day predictions contain non-finite prices")

    frames: dict[str, pd.DataFrame] = {}
    for result_type in EXPECTED_RESULT_TYPES:
        rows = selected[selected["result_type"] == result_type].sort_values(
            "prediction_date"
        )
        if len(rows) != EXPECTED_SAMPLES:
            raise EvidenceIntegrityError(
                f"{result_type} requires {EXPECTED_SAMPLES} test rows, found {len(rows)}"
            )
        if rows["prediction_date"].duplicated().any():
            raise EvidenceIntegrityError(f"duplicate test dates for {result_type}")
        frames[result_type] = rows.reset_index(drop=True)

    reference = frames[EXPECTED_RESULT_TYPES[0]]
    for result_type in EXPECTED_RESULT_TYPES[1:]:
        rows = frames[result_type]
        if not np.array_equal(
            rows["prediction_date"].to_numpy(), reference["prediction_date"].to_numpy()
        ):
            raise EvidenceIntegrityError("350-day checkpoint prediction dates are not aligned")
        if not np.array_equal(
            rows["target_close"].to_numpy(dtype=float),
            reference["target_close"].to_numpy(dtype=float),
        ):
            raise EvidenceIntegrityError("350-day checkpoint target prices are not aligned")
    if np.any(reference["target_close"].to_numpy(dtype=float) <= 0.0):
        raise EvidenceIntegrityError("350-day target prices must be positive for MAPE")
    return frames


def _paper_reference(evidence: FinalEvidence) -> dict[str, Any]:
    paper = evidence.read_csv("forecast/paper_reported_metrics.csv")
    required = {
        "model_id",
        "display_name",
        "RMSE",
        "MAE",
        "MAPE_pct",
        "source_label",
        "source_table",
        "source_url",
    }
    missing = required - set(paper.columns)
    if missing:
        raise EvidenceIntegrityError(
            f"paper reference columns missing: {sorted(missing)}"
        )
    selected = paper[paper["model_id"] == "cmamba_v"]
    if len(selected) != 1:
        raise EvidenceIntegrityError("paper reference requires exactly one cmamba_v row")
    row = selected.iloc[0]
    try:
        metrics = {name: float(row[name]) for name in ("RMSE", "MAE", "MAPE_pct")}
    except (TypeError, ValueError) as exc:
        raise EvidenceIntegrityError("paper CM-v metrics are invalid") from exc
    if not all(np.isfinite(value) and value > 0.0 for value in metrics.values()):
        raise EvidenceIntegrityError("paper CM-v metrics must be finite and positive")
    return {
        "model_id": "cmamba_v",
        "display_name": str(row["display_name"]),
        **metrics,
        "source_label": str(row["source_label"]),
        "source_table": str(row["source_table"]),
        "source_url": str(row["source_url"]),
    }


def build_reproduction_350d(evidence: FinalEvidence, core_root: Path) -> dict[str, Any]:
    """Return checksum-backed RQ1 metrics; never consume the legacy metrics CSV."""
    source_data, source_sha256, source_bytes = _pinned_source(evidence, core_root)
    frames = _load_test_predictions(source_data)
    paper = _paper_reference(evidence)
    date_reference = frames[EXPECTED_RESULT_TYPES[0]]["prediction_date"]

    rows: list[dict[str, Any]] = []
    display_names = {
        "official_checkpoint": "Official checkpoint",
        "retrained_checkpoint": "Reproduced checkpoint",
    }
    for result_type in EXPECTED_RESULT_TYPES:
        frame = frames[result_type]
        target = frame["target_close"].to_numpy(dtype=float)
        predicted = frame["predicted_close"].to_numpy(dtype=float)
        error = predicted - target
        metrics = {
            "RMSE": float(np.sqrt(np.mean(np.square(error)))),
            "MAE": float(np.mean(np.abs(error))),
            "MAPE_pct": float(np.mean(np.abs(error) / target) * 100.0),
        }
        gap_fields = {
            "RMSE": "RMSE_gap_pct",
            "MAE": "MAE_gap_pct",
            "MAPE_pct": "MAPE_gap_pct",
        }
        gaps = {
            gap_fields[metric]: abs(value - float(paper[metric]))
            / float(paper[metric])
            * 100.0
            for metric, value in metrics.items()
        }
        rows.append(
            {
                "result_type": result_type,
                "display_name": display_names[result_type],
                "samples": EXPECTED_SAMPLES,
                "date_from": date_reference.iloc[0].strftime("%Y-%m-%d"),
                "date_to": date_reference.iloc[-1].strftime("%Y-%m-%d"),
                **metrics,
                **gaps,
                "tolerance_pct": REPRODUCTION_THRESHOLD_PCT,
                "status": (
                    "PASS"
                    if all(gap <= REPRODUCTION_THRESHOLD_PCT for gap in gaps.values())
                    else "FAIL"
                ),
            }
        )

    return {
        "status": "READY",
        "error": None,
        "split": "test",
        "samples": EXPECTED_SAMPLES,
        "date_from": date_reference.iloc[0].strftime("%Y-%m-%d"),
        "date_to": date_reference.iloc[-1].strftime("%Y-%m-%d"),
        "criterion": {
            "threshold_pct": REPRODUCTION_THRESHOLD_PCT,
            "label": "Internal project reproduction criterion (not published by the paper).",
        },
        "paper_reference": paper,
        "rows": rows,
        "evidence": {
            "source_artifact": SOURCE_ARTIFACT,
            "source_sha256": source_sha256,
            "source_bytes": source_bytes,
            "manifest_artifact": SOURCE_MANIFEST,
            "evidence_package_sha256": evidence.package_sha256,
        },
    }


def not_ready_reproduction(error: str) -> dict[str, Any]:
    return {
        "status": "NOT_READY",
        "error": error,
        "split": "test",
        "samples": 0,
        "date_from": None,
        "date_to": None,
        "criterion": {
            "threshold_pct": REPRODUCTION_THRESHOLD_PCT,
            "label": "Internal project reproduction criterion (not published by the paper).",
        },
        "rows": [],
        "evidence": {},
    }


__all__ = ["build_reproduction_350d", "not_ready_reproduction"]
