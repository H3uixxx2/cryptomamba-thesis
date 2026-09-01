"""Directional accuracy + statistical significance tests (Task 2.14).

Compares CryptoMamba-v (retrained checkpoint) against every available baseline
on the **paper test split**, aligned by ``prediction_date`` (all share the same
350 dates 2023-10-01..2024-09-14).

For each baseline it reports:
  * directional accuracy of CryptoMamba-v and of the baseline
  * Diebold-Mariano test (squared-error loss, Newey-West lag-1 variance)
  * Wilcoxon signed-rank test on |error| differences

DM/Wilcoxon convention
----------------------
``d_t = e_cm^2 - e_base^2``. Mean(d) < 0 => CryptoMamba-v has the lower error =>
"better_model = CryptoMamba-v". ``significant`` is at alpha=0.05 (two-sided).
A NOT-significant result is reported honestly (it is a real RQ1/RQ2 finding,
not a failure to hide).

Baseline prediction sources (auto-detected; missing ones are skipped):
  * naive_persistence -> output/evaluation/baseline_predictions.csv
  * lstm/gru/itransformer -> output/evaluation/baseline_<model>_predictions.csv
    (drop neural per-date predictions there from Colab to extend this report)

Usage
-----
    .venv/bin/python scripts/significance_tests.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "evaluation"

ALPHA = 0.05
# Relative HOLD threshold: |pred-current| <= DIR_EPS_REL*current counts as no-direction.
# Kills float-noise sign flips (e.g. naive_persistence, whose pred==current exactly) without
# masking any real forecast move (neural-baseline moves are orders of magnitude larger).
DIR_EPS_REL = 1e-6

# baseline name -> (predictions csv, optional model-column filter value)
BASELINE_SOURCES: list[tuple[str, str, str | None]] = [
    ("naive_persistence", "baseline_predictions.csv", "naive_persistence"),
    ("lstm", "baseline_lstm_predictions.csv", None),
    ("gru", "baseline_gru_predictions.csv", None),
    ("itransformer", "baseline_itransformer_predictions.csv", None),
]


def _direction(delta: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Sign of a price move, with a relative HOLD band so float noise -> HOLD (0)."""
    eps = DIR_EPS_REL * np.abs(current)
    d = np.zeros_like(delta)
    d[delta > eps] = 1.0
    d[delta < -eps] = -1.0
    return d


def directional_stats(current: np.ndarray, target: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Return (strict directional accuracy %, directional coverage %).

    Coverage = share of rows where the model actually predicts a direction (non-HOLD).
    A pure-HOLD model (naive_persistence) has ~0% coverage, so its strict accuracy is
    not a real directional signal and must be read alongside coverage.
    """
    actual_dir = _direction(target - current, current)
    pred_dir = _direction(pred - current, current)
    strict = float(np.mean(actual_dir == pred_dir) * 100.0)
    coverage = float(np.mean(pred_dir != 0) * 100.0)
    return strict, coverage


def diebold_mariano(e_cm: np.ndarray, e_base: np.ndarray) -> tuple[float, float]:
    """DM stat + two-sided p using squared-error loss and Newey-West lag-1 var."""
    d = e_cm**2 - e_base**2
    n = len(d)
    d_bar = float(np.mean(d))
    gamma0 = float(np.mean((d - d_bar) ** 2))
    gamma1 = float(np.mean((d[1:] - d_bar) * (d[:-1] - d_bar)))
    var_d = (gamma0 + 2.0 * gamma1) / n
    if var_d <= 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var_d)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return round(float(dm), 4), round(float(p), 4)


def load_cm_retrained_test() -> pd.DataFrame:
    fc = pd.read_csv(OUTPUT_DIR / "forecast_predictions.csv")
    cm = fc[(fc["result_type"] == "retrained_checkpoint") & (fc["split"] == "test")]
    cols = ["prediction_date", "current_close", "target_close", "predicted_close"]
    return cm[cols].sort_values("prediction_date").reset_index(drop=True)


def load_baseline(filename: str, model_filter: str | None) -> pd.DataFrame | None:
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    bp = pd.read_csv(path)
    if model_filter is not None and "model" in bp.columns:
        bp = bp[bp["model"] == model_filter]
    if "split" in bp.columns:
        bp = bp[bp["split"] == "test"]
    cols = ["prediction_date", "current_close", "predicted_close"]
    if not set(cols).issubset(bp.columns):
        return None
    return bp[cols].sort_values("prediction_date").reset_index(drop=True)


def compare(cm: pd.DataFrame, name: str, base: pd.DataFrame) -> dict[str, Any] | None:
    # CryptoMamba-v carries the canonical actual prices (current_close/target_close).
    # Each model's directional prediction is judged against ITS OWN current_close to avoid
    # cross-file float-noise sign flips.
    merged = cm.merge(base, on="prediction_date", suffixes=("_cm", "_base"))
    if merged.empty:
        print(f"  [skip] {name}: no overlapping prediction_date with CryptoMamba-v")
        return None
    cm_current = merged["current_close_cm"].to_numpy(float)
    target = merged["target_close"].to_numpy(float)
    cm_pred = merged["predicted_close_cm"].to_numpy(float)
    base_current = merged["current_close_base"].to_numpy(float)
    base_pred = merged["predicted_close_base"].to_numpy(float)

    e_cm = cm_pred - target
    e_base = base_pred - target

    dm_stat, dm_p = diebold_mariano(e_cm, e_base)
    diff = np.abs(e_cm) - np.abs(e_base)
    if np.allclose(diff, 0):
        w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = stats.wilcoxon(diff)
        w_stat, w_p = round(float(w_stat), 4), round(float(w_p), 4)

    cm_da, cm_cov = directional_stats(cm_current, target, cm_pred)
    base_da, base_cov = directional_stats(base_current, target, base_pred)

    # mean(d)<0 => CM lower squared error => CM better
    mean_d = float(np.mean(e_cm**2 - e_base**2))
    better = "CryptoMamba-v" if mean_d < 0 else name
    significant = bool(np.isfinite(dm_p) and dm_p < ALPHA)

    print(
        f"  {name}: DA cm={cm_da:.2f}%(cov {cm_cov:.0f}%) base={base_da:.2f}%(cov {base_cov:.0f}%) | "
        f"DM stat={dm_stat} p={dm_p} | Wilcoxon p={w_p} | "
        f"lower_MSE={better} significant={significant}"
    )
    return {
        "comparison": f"CryptoMamba-v vs {name}",
        "samples": int(len(merged)),
        "cm_directional_acc_pct": round(cm_da, 4),
        "cm_directional_coverage_pct": round(cm_cov, 4),
        "baseline_directional_acc_pct": round(base_da, 4),
        "baseline_directional_coverage_pct": round(base_cov, 4),
        "dm_statistic": dm_stat,
        "dm_p_value": dm_p,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_p,
        "lower_mse_model": better,
        "conclusion": "significant" if significant else "not_significant",
    }


def main() -> None:
    cm = load_cm_retrained_test()
    print(f"CryptoMamba-v retrained test rows: {len(cm)}")

    rows: list[dict[str, Any]] = []
    for name, filename, model_filter in BASELINE_SOURCES:
        base = load_baseline(filename, model_filter)
        if base is None:
            print(f"  [skip] {name}: {filename} not found")
            continue
        result = compare(cm, name, base)
        if result is not None:
            rows.append(result)

    if not rows:
        raise RuntimeError("No baseline comparisons produced; check prediction artifacts")

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "significance_tests.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
