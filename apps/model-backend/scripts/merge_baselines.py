"""Merge all available baselines into one forecast-comparison table (Task 2.13/2.15).

Auto-discovers per-date prediction files under output/evaluation/ and recomputes every model's
test metrics with ONE consistent formula set (so all rows are directly comparable),
then appends the CryptoMamba-v rows from forecast_metrics.csv and writes:

    output/evaluation/baseline_metrics_comparison.csv

Non-destructive: it does NOT overwrite baseline_metrics.csv (the naive train/val/test artifact the
UI currently reads). Wiring this comparison into the Reproduce screen + flipping it to READY is the
deliberate Task 2.15 step, done once all four baselines exist.

Discovered sources (missing ones are skipped, and logged):
    naive_persistence -> baseline_predictions.csv
    lstm/gru/itransformer -> baseline_<model>_predictions.csv

Usage:
    .venv/bin/python scripts/merge_baselines.py    # local; re-run after neural preds arrive
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "evaluation"
DIR_EPS_REL = 1e-6

# model -> (predictions file, model-column filter, paper RMSE, paper MAPE_pct)
SOURCES: list[tuple[str, str, str | None, float | None, float | None]] = [
    ("naive_persistence", "baseline_predictions.csv", "naive_persistence", None, None),
    ("lstm", "baseline_lstm_predictions.csv", None, 2202.1, 2.896),
    ("gru", "baseline_gru_predictions.csv", None, 1978.0, 2.526),
    ("itransformer", "baseline_itransformer_predictions.csv", None, 1779.9, 2.427),
]
PAPER_CM = (1598.1, 2.034)  # CryptoMamba-v paper target


def _direction(delta: np.ndarray, current: np.ndarray) -> np.ndarray:
    eps = DIR_EPS_REL * np.abs(current)
    d = np.zeros_like(delta)
    d[delta > eps] = 1.0
    d[delta < -eps] = -1.0
    return d


def _metrics_from_predictions(df: pd.DataFrame) -> dict[str, Any]:
    cur = df["current_close"].to_numpy(float)
    tgt = df["target_close"].to_numpy(float)
    prd = df["predicted_close"].to_numpy(float)
    err = prd - tgt
    pred_dir = _direction(prd - cur, cur)
    actual_dir = _direction(tgt - cur, cur)
    return {
        "samples": int(len(df)),
        "RMSE": round(float(np.sqrt(np.mean(err**2))), 6),
        "MAE": round(float(np.mean(np.abs(err))), 6),
        "MAPE_pct": round(float(np.mean(np.abs(err / tgt)) * 100.0), 6),
        "directional_accuracy_strict_pct": round(float(np.mean(actual_dir == pred_dir) * 100.0), 6),
        "directional_coverage_pct": round(float(np.mean(pred_dir != 0) * 100.0), 6),
    }


def _gap_pct(value: float, paper: float | None) -> float:
    if paper is None or paper == 0:
        return math.nan
    return round(abs(value - paper) / paper * 100.0, 4)


def main() -> None:
    rows: list[dict[str, Any]] = []
    found, missing = [], []

    for name, filename, model_filter, paper_rmse, paper_mape in SOURCES:
        path = OUTPUT_DIR / filename
        if not path.exists():
            missing.append(name)
            continue
        df = pd.read_csv(path)
        if model_filter is not None and "model" in df.columns:
            df = df[df["model"] == model_filter]
        if "split" in df.columns:
            df = df[df["split"] == "test"]
        if df.empty:
            missing.append(name)
            continue
        m = _metrics_from_predictions(df)
        rows.append(
            {
                "model": name,
                "split": "test",
                **m,
                "paper_RMSE": paper_rmse if paper_rmse is not None else math.nan,
                "paper_MAPE_pct": paper_mape if paper_mape is not None else math.nan,
                "RMSE_gap_pct": _gap_pct(m["RMSE"], paper_rmse),
                "MAPE_gap_pct": _gap_pct(m["MAPE_pct"], paper_mape),
            }
        )
        found.append(name)

    # Append CryptoMamba-v (official + retrained) from the existing forecast artifact.
    fm_path = OUTPUT_DIR / "forecast_metrics.csv"
    fp_path = OUTPUT_DIR / "forecast_predictions.csv"
    # Directional accuracy for both CryptoMamba-v checkpoints, computed from the
    # per-date predictions with the same strict rule used in significance_tests.py.
    cm_dir: dict[str, tuple[float, float]] = {}
    if fp_path.exists():
        fp = pd.read_csv(fp_path)
        for rt, grp in fp[fp["split"] == "test"].groupby("result_type"):
            actual_dir = np.sign(grp["target_close"].to_numpy() - grp["current_close"].to_numpy())
            pred_dir = np.sign(grp["predicted_close"].to_numpy() - grp["current_close"].to_numpy())
            cm_dir[str(rt)] = (
                round(float(np.mean(actual_dir == pred_dir) * 100.0), 6),
                round(float(np.mean(pred_dir != 0) * 100.0), 6),
            )
    if fm_path.exists():
        fm = pd.read_csv(fm_path)
        for _, r in fm[fm["split"] == "test"].iterrows():
            tag = "CryptoMamba-v (official)" if r["result_type"] == "official_checkpoint" else "CryptoMamba-v (retrained)"
            d_acc, d_cov = cm_dir.get(str(r["result_type"]), (math.nan, math.nan))
            rows.append(
                {
                    "model": tag,
                    "split": "test",
                    "samples": int(r["samples"]),
                    "RMSE": round(float(r["RMSE"]), 6),
                    "MAE": round(float(r["MAE"]), 6),
                    "MAPE_pct": round(float(r["MAPE_pct"]), 6),
                    "directional_accuracy_strict_pct": d_acc,
                    "directional_coverage_pct": d_cov,
                    "paper_RMSE": PAPER_CM[0],
                    "paper_MAPE_pct": PAPER_CM[1],
                    "RMSE_gap_pct": _gap_pct(float(r["RMSE"]), PAPER_CM[0]),
                    "MAPE_gap_pct": _gap_pct(float(r["MAPE_pct"]), PAPER_CM[1]),
                }
            )

    if not rows:
        raise RuntimeError("No baseline predictions found under output/evaluation/")

    out = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    out_path = OUTPUT_DIR / "baseline_metrics_comparison.csv"
    out.to_csv(out_path, index=False)

    print(f"Found baselines: {found}")
    if missing:
        print(f"Still missing (Colab GPU): {missing}")
    print(f"\nWrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
