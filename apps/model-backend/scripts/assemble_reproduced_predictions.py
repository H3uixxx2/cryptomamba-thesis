"""Assemble the reproduced-CM-v rows of the frozen evaluation artifacts from a seed run.

The official checkpoint's rows in ``output/evaluation`` are never touched. The
``retrained_checkpoint`` rows are rebuilt from the per-date predictions stored with the run
(``output/seed_runs/<run>/{val,test}_preds.csv``):

  forecast_predictions.csv     val + test rows (350 dates each); no train rows exist for the run
  forecast_metrics.csv         350-date test metrics and gaps to the paper
  trading_replay_metrics.csv   the paper's released strategies, zero fee, on those predictions

Nothing is estimated and no model is loaded. Each run's prediction file carries its own price
columns, so the script also checks that they equal the frozen price base date by date.

Usage (from apps/model-backend):
    python scripts/assemble_reproduced_predictions.py
    python scripts/assemble_reproduced_predictions.py --run cmamba_v__seed23
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from utils.trade import trade  # noqa: E402

EVALUATION_DIR = CORE_ROOT / "output" / "evaluation"
RUNS_DIR = CORE_ROOT / "output" / "seed_runs"
SPLIT_DATA = CORE_ROOT / "data" / "2018-09-17_2024-09-16_86400"
REDRAWN = "retrained_checkpoint"
OFFICIAL = "official_checkpoint"
SPLITS = ("val", "test")
MODES = ("vanilla", "smart", "smart_w_short")
INITIAL_BALANCE = 100.0
RISK_PCT = 2.0
PRICE_TOLERANCE = 1e-3
VERIFIED_TOLERANCE = 0.05  # the paper prints balances and drawdowns to two decimals
FORK_COMMIT = "672faa9fb7cb4499da17d6a3afbd65a9301f170f"
PAPER_TEST = {"RMSE": 1598.1, "MAE": 1120.7, "MAPE_pct": 2.034}
TOLERANCE_PCT = 5.0


def _maximum_drawdown(values) -> float:
    values = np.asarray(values, dtype=float)
    peaks = np.maximum.accumulate(values)
    return float(-np.min((values - peaks) / peaks) * 100.0)


def _load_run(run_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / f"{split}_preds.csv")
    missing = {"timestamp", "y", "y_hat", "y_old"} - set(frame.columns)
    if missing:
        raise ValueError(f"{run_dir.name}/{split}_preds.csv: missing columns {sorted(missing)}")
    frame["timestamp"] = frame["timestamp"].astype("int64")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{run_dir.name}/{split}_preds.csv: duplicate timestamps")
    if not np.isfinite(frame[["y", "y_hat", "y_old"]].to_numpy(float)).all():
        raise ValueError(f"{run_dir.name}/{split}_preds.csv: non-finite values")
    return frame


def _checkpoint(run_dir: Path) -> Path:
    found = sorted(run_dir.glob("*.ckpt"))
    if len(found) != 1:
        raise ValueError(f"{run_dir}: expected exactly one checkpoint, found {len(found)}")
    return found[0]


def _reproduced_rows(base: pd.DataFrame, run_dir: Path, checkpoint_rel: str) -> pd.DataFrame:
    """Rows for one run: the official rows are the date/price template."""
    frames = []
    for split in SPLITS:
        template = base[(base["result_type"] == OFFICIAL) & (base["split"] == split)]
        template = template.sort_values("sample_index").reset_index(drop=True)
        run = _load_run(run_dir, split)
        merged = template.merge(
            run, left_on="prediction_timestamp", right_on="timestamp", how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(template):
            raise ValueError(f"{split}: {len(merged)} of {len(template)} dates found in the run")
        if not np.allclose(merged["y"], merged["target_close"], rtol=0.0, atol=PRICE_TOLERANCE):
            raise ValueError(f"{split}: run target prices differ from the frozen price base")
        if not np.allclose(merged["y_old"], merged["current_close"], rtol=0.0, atol=PRICE_TOLERANCE):
            raise ValueError(f"{split}: run current prices differ from the frozen price base")
        rows = template.copy()
        rows["result_type"] = REDRAWN
        rows["checkpoint"] = checkpoint_rel
        rows["source_commit"] = FORK_COMMIT
        rows["predicted_close"] = merged["y_hat"].to_numpy(float)
        rows["predicted_return_pct"] = (
            (rows["predicted_close"] / rows["current_close"] - 1.0) * 100.0
        )
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def _metrics_row(rows: pd.DataFrame, previous: pd.Series, checkpoint_rel: str) -> dict:
    test = rows[rows["split"] == "test"].sort_values("sample_index")
    error = test["predicted_close"].to_numpy(float) - test["target_close"].to_numpy(float)
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    mape = float(np.mean(np.abs(error / test["target_close"].to_numpy(float))) * 100.0)
    gaps = {
        key: abs(value - PAPER_TEST[key]) / PAPER_TEST[key] * 100.0
        for key, value in (("RMSE", rmse), ("MAE", mae), ("MAPE_pct", mape))
    }
    row = previous.to_dict()
    row.update(
        checkpoint=checkpoint_rel,
        source_commit=FORK_COMMIT,
        samples=int(len(test)),
        RMSE=rmse,
        MAE=mae,
        MAPE_pct=mape,
        RMSE_gap_pct=gaps["RMSE"],
        MAE_gap_pct=gaps["MAE"],
        MAPE_gap_pct=gaps["MAPE_pct"],
        status="PASS" if max(gaps.values()) < TOLERANCE_PCT else "FAIL",
    )
    return row


def _replay_rows(rows: pd.DataFrame, previous: pd.DataFrame, checkpoint_rel: str) -> pd.DataFrame:
    updated = previous.copy()
    for split in SPLITS:
        frame = rows[rows["split"] == split].sort_values("sample_index")
        raw = pd.read_csv(SPLIT_DATA / f"{split}.csv", index_col=0).reset_index(drop=True)
        for mode in MODES:
            final, curve = trade(
                raw.copy(), "Timestamp",
                frame["prediction_timestamp"].astype(int).tolist(),
                frame["target_close"].to_numpy(float),
                frame["predicted_close"].to_numpy(float),
                balance=INITIAL_BALANCE, mode=mode, risk=RISK_PCT, y_key="Close",
            )
            drawdown = _maximum_drawdown(curve)
            mask = (
                (updated["result_type"] == REDRAWN)
                & (updated["split"] == split)
                & (updated["trade_mode"] == mode)
            )
            if int(mask.sum()) != 1:
                raise ValueError(f"replay row for {split}/{mode} is not unique")
            index = updated.index[mask][0]
            paper_final = float(updated.at[index, "paper_final_balance"])
            paper_drawdown = float(updated.at[index, "paper_max_drawdown_pct"])
            balance_gap = (float(final) - paper_final) / paper_final * 100.0
            drawdown_gap = drawdown - paper_drawdown
            updated.loc[index, "checkpoint"] = checkpoint_rel
            updated.loc[index, "source_commit"] = FORK_COMMIT
            updated.loc[index, "final_balance"] = float(final)
            updated.loc[index, "max_drawdown_pct"] = drawdown
            updated.loc[index, "balance_gap_pct"] = balance_gap
            updated.loc[index, "mdd_gap_pp"] = drawdown_gap
            matched = (
                abs(float(final) - paper_final) <= VERIFIED_TOLERANCE
                and abs(drawdown_gap) <= VERIFIED_TOLERANCE
            )
            updated.loc[index, "status"] = "VERIFIED" if matched else "NOT_MATCHED"
    return updated


def assemble(run: str) -> dict:
    run_dir = RUNS_DIR / run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory is missing: {run_dir}")
    checkpoint = _checkpoint(run_dir)
    checkpoint_rel = checkpoint.relative_to(CORE_ROOT).as_posix()

    predictions_path = EVALUATION_DIR / "forecast_predictions.csv"
    metrics_path = EVALUATION_DIR / "forecast_metrics.csv"
    replay_path = EVALUATION_DIR / "trading_replay_metrics.csv"
    base = pd.read_csv(predictions_path)
    columns = list(base.columns)
    official = base[base["result_type"] == OFFICIAL]
    if official.empty:
        raise ValueError("forecast_predictions.csv has no official checkpoint rows")

    reproduced = _reproduced_rows(base, run_dir, checkpoint_rel)
    output = pd.concat([official, reproduced[columns]], ignore_index=True)
    output.to_csv(predictions_path, index=False, lineterminator="\n")

    metrics = pd.read_csv(metrics_path)
    mask = (metrics["result_type"] == REDRAWN) & (metrics["split"] == "test")
    if int(mask.sum()) != 1:
        raise ValueError("forecast_metrics.csv must hold exactly one reproduced test row")
    index = metrics.index[mask][0]
    row = _metrics_row(reproduced, metrics.loc[index], checkpoint_rel)
    for key, value in row.items():
        metrics.loc[index, key] = value
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")

    replay = _replay_rows(reproduced, pd.read_csv(replay_path), checkpoint_rel)
    replay.to_csv(replay_path, index=False, lineterminator="\n")
    return {
        "run": run,
        "checkpoint": checkpoint_rel,
        "prediction_rows": int(len(output)),
        "test_rmse": float(row["RMSE"]),
        "test_mae": float(row["MAE"]),
        "test_mape_pct": float(row["MAPE_pct"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="cmamba_v__seed23",
                        help="Directory name under output/seed_runs (default: cmamba_v__seed23)")
    args = parser.parse_args()
    print(json.dumps(assemble(args.run), indent=2))


if __name__ == "__main__":
    main()
