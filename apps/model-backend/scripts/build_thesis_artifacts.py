"""Build the deterministic final-thesis evaluation artifacts.

This command reads frozen predictions/checkpoints only.  It never trains a
model and never rewrites the original paper replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd

CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from thesis_pipeline.backtest import SelfFinancingConfig, run_corrected_matrix
from thesis_pipeline.contracts import MODEL_SPECS
from thesis_pipeline.evaluation import (
    build_aligned_predictions,
    forecast_metrics,
    paired_tests,
)


OUTPUT_FILES = (
    "paper_reported_metrics.csv",
    "controlled_predictions.csv",
    "controlled_forecast_metrics.csv",
    "paired_significance_tests.csv",
    "paper_replay_metrics.csv",
    "corrected_trading_metrics.csv",
    "corrected_trading_equity.csv",
    "corrected_trading_metadata.json",
    "s5_full_summary.json",
    "checkpoint_provenance.json",
    "artifact_manifest.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _checkpoint_provenance(core_root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_id, spec in MODEL_SPECS.items():
        entry: dict[str, Any] = {
            "display_name": spec.display_name,
            "window_days": spec.window_days,
            "parameter_count": spec.parameter_count,
            "prediction_mode": spec.prediction_mode,
            "source_commit": spec.source_commit,
            "checkpoint_path": (
                str(spec.checkpoint_path) if spec.checkpoint_path is not None else None
            ),
            "checkpoint_sha256": spec.checkpoint_sha256,
        }
        if spec.checkpoint_path is not None:
            checkpoint = core_root / spec.checkpoint_path
            if not checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
            actual = _sha256(checkpoint)
            if actual != spec.checkpoint_sha256:
                raise ValueError(
                    f"checkpoint hash mismatch for {model_id}: "
                    f"expected {spec.checkpoint_sha256}, got {actual}"
                )
            entry["checkpoint_bytes"] = checkpoint.stat().st_size
            entry["hash_verified"] = True
        else:
            entry["checkpoint_bytes"] = 0
            entry["hash_verified"] = False
        models[model_id] = entry
    return {"schema_version": 1, "models": models}


def _file_manifest(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        entry["rows"] = int(len(frame))
        entry["columns"] = frame.columns.tolist()
    return entry


def build_all(*, core_root: Path, output_dir: Path) -> dict[str, Path]:
    core_root = Path(core_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = build_aligned_predictions(core_root)
    metrics = forecast_metrics(predictions)
    significance = paired_tests(predictions)
    configs = [
        SelfFinancingConfig(
            initial_equity=100.0,
            transaction_cost_pct=cost,
            borrow_cost_bps_per_day=0.0,
            risk_pct=2.0,
            max_long_exposure=1.0,
            max_short_exposure=1.0,
            gross_exposure_cap=1.0,
        )
        for cost in (0.0, 0.1, 0.2)
    ]
    corrected = run_corrected_matrix(predictions, configs=configs)

    paths = {name: output_dir / name for name in OUTPUT_FILES}
    paper_source = (
        core_root / "thesis_pipeline/data/paper_reported_metrics.csv"
    )
    paper_replay_source = core_root / "output/evaluation/trading_replay_metrics.csv"
    s5_summary_source = (
        core_root
        / "output/improve_track_evidence/s5_full/s5_full_summary.json"
    )
    for source in (paper_source, paper_replay_source, s5_summary_source):
        if not source.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {source}")

    shutil.copyfile(paper_source, paths["paper_reported_metrics.csv"])
    _write_csv(paths["controlled_predictions.csv"], predictions)
    _write_csv(paths["controlled_forecast_metrics.csv"], metrics)
    _write_csv(paths["paired_significance_tests.csv"], significance)
    shutil.copyfile(paper_replay_source, paths["paper_replay_metrics.csv"])
    _write_csv(paths["corrected_trading_metrics.csv"], corrected["metrics"])
    _write_csv(paths["corrected_trading_equity.csv"], corrected["equity"])

    corrected_metadata = {
        "schema_version": 1,
        **corrected["metadata"],
        "model_ids": list(MODEL_SPECS),
        "splits": ["val", "test"],
        "transaction_cost_pct_scenarios": [0.0, 0.1, 0.2],
        "thesis_reference_cost_pct": 0.2,
        "initial_equity": 100.0,
        "risk_pct": 2.0,
        "borrow_cost_bps_per_day": 0.0,
        "borrow_cost_note": "Parameterised; the frozen thesis matrix uses zero daily borrow cost.",
        "same_close_execution": True,
        "signal_information": ["current_close", "predicted_close"],
        "target_close_used_only_for": ["mark_to_market", "outcome_metrics"],
        "controlled_rows_per_model_split": 304,
    }
    _write_json(paths["corrected_trading_metadata.json"], corrected_metadata)
    shutil.copyfile(s5_summary_source, paths["s5_full_summary.json"])
    _write_json(
        paths["checkpoint_provenance.json"], _checkpoint_provenance(core_root)
    )

    source_artifacts = {
        str(path.relative_to(core_root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (
            core_root / "output/evaluation/forecast_predictions.csv",
            core_root
            / "output/improve_track_evidence/s5_full/preds/s5_full__seed23__val.csv",
            core_root
            / "output/improve_track_evidence/s5_full/preds/s5_full__seed23__test.csv",
            paper_replay_source,
            s5_summary_source,
        )
    }
    generated_names = [
        name for name in OUTPUT_FILES if name != "artifact_manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "authority_order": [
            "original paper and official source",
            "frozen checkpoints and per-date predictions",
            "deterministic recomputation",
            "checksum-verified evidence copy",
            "thesis and UI presentation",
        ],
        "invariants": {
            "approved_model_ids": list(MODEL_SPECS),
            "controlled_rows_per_model_split": 304,
            "dm_loss": "squared_error",
            "dm_hac_lag": 1,
            "wilcoxon_loss": "absolute_error",
            "paper_replay_preserved": True,
            "corrected_engine": "self_financing_same_close_v1",
        },
        "source_artifacts": source_artifacts,
        "files": {
            name: _file_manifest(paths[name]) for name in generated_names
        },
    }
    _write_json(paths["artifact_manifest.json"], manifest)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen forecast/statistical/corrected-trading thesis artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: <core>/output/thesis_final).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = CORE_ROOT
    destination = args.output_dir or root / "output/thesis_final"
    built = build_all(core_root=root, output_dir=destination)
    metrics = pd.read_csv(built["corrected_trading_metrics.csv"])
    predictions = pd.read_csv(built["controlled_predictions.csv"])
    print(
        f"Built {len(built)} artifacts in {Path(destination).resolve()} "
        f"({len(predictions)} controlled prediction rows; "
        f"{len(metrics)} corrected trading rows)."
    )
    print(
        "Manifest SHA-256: " + _sha256(built["artifact_manifest.json"])
    )
