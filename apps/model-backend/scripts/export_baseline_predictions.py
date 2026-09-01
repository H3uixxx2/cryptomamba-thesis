"""Export per-date TEST predictions for a trained NN baseline (LSTM/GRU/iTransformer).

Runs on Colab GPU right after ``scripts/training.py``. For one model it writes:
  output/evaluation/baseline_<name>_predictions.csv  (per-date; naive-compatible schema)
  output/evaluation/baseline_<name>_metrics.csv       (test RMSE/MAE/MAPE_pct + directional acc)

Why this exists
---------------
``scripts/evaluation.py`` only *prints* aggregate metrics; ``.ai/phase-2-baseline-impl.md``
tells you to copy them by hand. But Task 2.14 (Diebold-Mariano / Wilcoxon / directional accuracy)
needs **per-date** baseline predictions aligned with CryptoMamba-v. This script produces them by
mirroring ``evaluation.py``'s exact model/data/denormalization path (same ``DataTransform``,
``CMambaDataModule``, factors-based min-max denorm, ``model(features)`` forward), so neural-baseline
numbers are computed under the **identical protocol** as CryptoMamba-v. The output drops straight
into ``scripts/significance_tests.py`` (auto-detected) and the Task 2.13 merge.

Dates use ``pd.to_datetime(ts, unit="s")`` (UTC) to match the naive/CryptoMamba prediction
dates exactly, so the significance merge on ``prediction_date`` aligns row-for-row (350 rows,
2023-10-01..2024-09-14).

Usage (Colab)
-------------
    python scripts/export_baseline_predictions.py --config lstm_v \
        --ckpt_path logs/lstm/version_0/checkpoints/<best>.ckpt \
        --model_name lstm --use_volume
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from data_utils.data_transforms import DataTransform
from pl_modules.data_module import CMambaDataModule
from utils import io_tools

warnings.simplefilter(action="ignore", category=FutureWarning)

ROOT = io_tools.get_root(__file__, num_returns=2)
OUTPUT_DIR = os.path.join(str(ROOT), "output", "evaluation")
DIR_EPS_REL = 1e-6  # HOLD band so float noise -> no-direction (matches significance_tests.py)


def get_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="training config name, e.g. lstm_v")
    p.add_argument("--ckpt_path", required=True, help="best checkpoint from training.py")
    p.add_argument("--model_name", required=True, help="row label: lstm / gru / itransformer")
    p.add_argument("--use_volume", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def load_model(config, ckpt_path):
    """Identical resolution to evaluation.py.load_model (+ CPU fallback)."""
    arch_config = io_tools.load_config_from_yaml("configs/models/archs.yaml")
    model_arch = config.get("model")
    model_config_path = f"{ROOT}/configs/models/{arch_config.get(model_arch)}"
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get("normalize", False)
    model_class = io_tools.get_obj_from_str(model_config.get("target"))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get("params"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, normalize, device


@torch.no_grad()
def collect(model, dataloader, device):
    ts, cur, tgt, prd = [], [], [], []
    for batch in dataloader:
        ts += [float(x) for x in batch.get("Timestamp").numpy().reshape(-1)]
        tgt += [float(x) for x in batch.get(model.y_key).numpy().reshape(-1)]
        cur += [float(x) for x in batch.get(f"{model.y_key}_old").numpy().reshape(-1)]
        feats = batch.get("features").to(device)
        prd += [float(x) for x in model(feats).cpu().numpy().reshape(-1)]
    return (np.asarray(ts), np.asarray(cur), np.asarray(tgt), np.asarray(prd))


def _denorm(arr, factors, key):
    scale = factors.get(key).get("max") - factors.get(key).get("min")
    shift = factors.get(key).get("min")
    return arr * scale + shift


def _direction(delta, current):
    eps = DIR_EPS_REL * np.abs(current)
    d = np.zeros_like(delta)
    d[delta > eps] = 1.0
    d[delta < -eps] = -1.0
    return d


def main():
    args = get_args()
    pl.seed_everything(args.seed)

    config = io_tools.load_config_from_yaml(f"{ROOT}/configs/training/{args.config}.yaml")
    data_config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml"
    )
    use_volume = args.use_volume or config.get("use_volume")
    extra = config.get("additional_features", [])
    transforms = {
        "train_transform": DataTransform(is_train=True, use_volume=use_volume, additional_features=extra),
        "val_transform": DataTransform(is_train=False, use_volume=use_volume, additional_features=extra),
        "test_transform": DataTransform(is_train=False, use_volume=use_volume, additional_features=extra),
    }

    model, normalize, device = load_model(config, args.ckpt_path)
    dm = CMambaDataModule(
        data_config,
        batch_size=args.batch_size,
        distributed_sampler=False,
        num_workers=args.num_workers,
        normalize=normalize,
        **transforms,
    )
    test_loader = dm.test_dataloader()
    factors = dm.factors if normalize else None

    ts, cur, tgt, prd = collect(model, test_loader, device)
    if factors is not None:
        cur = _denorm(cur, factors, model.y_key)
        tgt = _denorm(tgt, factors, model.y_key)
        prd = _denorm(prd, factors, model.y_key)
        ts = _denorm(ts, factors, "Timestamp")

    order = np.argsort(ts)
    ts, cur, tgt, prd = ts[order], cur[order], tgt[order], prd[order]
    dates = pd.to_datetime(ts.astype("int64"), unit="s").strftime("%Y-%m-%d")

    preds_df = pd.DataFrame(
        {
            "model": args.model_name,
            "split": "test",
            "sample_index": np.arange(len(ts)),
            "prediction_date": dates,
            "prediction_timestamp": ts.astype("int64"),
            "current_close": cur,
            "target_close": tgt,
            "predicted_close": prd,
            "actual_return_pct": (tgt - cur) / cur * 100.0,
            "predicted_return_pct": (prd - cur) / cur * 100.0,
            "absolute_error": np.abs(prd - tgt),
            "squared_error": (prd - tgt) ** 2,
        }
    )

    err = prd - tgt
    pred_dir = _direction(prd - cur, cur)
    actual_dir = _direction(tgt - cur, cur)
    metrics = {
        "model": args.model_name,
        "split": "test",
        "samples": int(len(ts)),
        "RMSE": round(float(np.sqrt(np.mean(err**2))), 6),
        "MAE": round(float(np.mean(np.abs(err))), 6),
        "MAPE_pct": round(float(np.mean(np.abs(err / tgt)) * 100.0), 6),
        "directional_accuracy_strict_pct": round(float(np.mean(actual_dir == pred_dir) * 100.0), 6),
        "directional_coverage_pct": round(float(np.mean(pred_dir != 0) * 100.0), 6),
        "checkpoint": os.path.basename(args.ckpt_path),
        "protocol": (
            f"{args.config}; seed={args.seed}; evaluation.py-equivalent inference; "
            f"normalize={normalize}; window=14; released sample count"
        ),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preds_path = os.path.join(OUTPUT_DIR, f"baseline_{args.model_name}_predictions.csv")
    metrics_path = os.path.join(OUTPUT_DIR, f"baseline_{args.model_name}_metrics.csv")
    preds_df.to_csv(preds_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"\n[{args.model_name}] test metrics:")
    for k in ("samples", "RMSE", "MAE", "MAPE_pct", "directional_accuracy_strict_pct", "directional_coverage_pct"):
        print(f"  {k}: {metrics[k]}")
    print(f"Wrote {preds_path} ({len(ts)} rows)")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
