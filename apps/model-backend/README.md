# model-backend — CryptoMamba research core

The model, its training entry point, the offline evaluation/trading engine, and the frozen artifacts they produce.

## Models

| Config | Model | Window | Parameters | Target |
|---|---|---|---|---|
| `cmamba_v` → `CMamba_v2` → `configs/models/CryptoMamba/v2.yaml` | CryptoMamba-v (`models/cmamba.py`) | 14 days | 136,952 | next-day Close, raw price |
| `s5_full` → `CMambaT_w60` → `configs/models/CryptoMamba/t2.yaml` | CryptoMamba-T (`models/cmamba_t.py`) | 60 days | 57,249 | next-day Close via relative return |

Both are wrapped by `pl_modules/base_module.py` (train/validation/test steps, price↔return reconstruction, denormalisation).
`configs/models/archs.yaml` maps the architecture name in a training config to its model YAML, and `utils/io_tools.py`
instantiates the class named in that YAML's `target:` field, so the Lightning wrappers are reached by string import.

`CMambaT` runs the selective scan along the day tokens instead of the six feature tokens.

## Layout

| Path | What |
|---|---|
| `models/` | `cmamba.py` (CryptoMamba-v + the pure-PyTorch `selective_scan_ref` CPU fallback), `cmamba_t.py` |
| `pl_modules/` | `base_module.py` (shared steps), `cmamba_module.py`, `cmamba_t_module.py`, `data_module.py` |
| `data_utils/` | `data_transforms.py` (feature tensor + hygiene flags), `dataset.py` (split cache reader) |
| `utils/` | `trade.py` (the paper's strategy logic), `io_tools.py` |
| `thesis_pipeline/` | offline engine over frozen checkpoints: `contracts`, `inference`, `evaluation`, `backtest`, `package` |
| `scripts/` | `training.py`, `evaluation.py`, `run_backtest.py`, `checkpoint_inference.py` (stdin/stdout JSON worker), `build_thesis_artifacts.py`, `package_thesis_evidence.py` |
| `configs/` | `data_configs/mode_1.yaml` (chronological split), `models/CryptoMamba/{v2,t2}.yaml`, `models/archs.yaml`, `training/{cmamba_v,s5_full}.yaml` |
| `checkpoints/cmamba_v.ckpt` | the official released checkpoint |
| `data/2018-09-17_2024-09-16_86400/` | the OHLCV split cache: 1461 / 365 / 365 rows |
| `output/` | frozen results, see below |

### Data

`data_utils/dataset.py::get_data` returns the tracked `train.csv` / `val.csv` / `test.csv` directly and falls back to
`data_path` only when that cache is absent. Splits are chronological:

| Split | Range | Rows |
|---|---|---|
| train | 2018-09-17 → 2022-09-16 | 1461 |
| validation | 2022-09-17 → 2023-09-16 | 365 |
| test | 2023-09-17 → 2024-09-15 | 365 |

## `output/`

| Path | Contents | Read by |
|---|---|---|
| `evaluation/forecast_predictions.csv` · `forecast_metrics.csv` | Per-date CM-v predictions and metrics for the official checkpoint and the reproduced (seed-23) run; the reproduced rows are written by `scripts/assemble_reproduced_predictions.py` from `seed_runs/cmamba_v__seed23/` | `thesis_pipeline`, `run_backtest.py`, console Evaluation |
| `evaluation/trading_replay_metrics.csv` | Paper replay | `run_backtest.py`, console Trading |
| `evaluation/trading_metrics.csv` · `trading_equity_curve.csv` · `regime_metrics.csv` · `trading_backtest_metadata.json` | Chronological backtest written by `run_backtest.py`; untracked | console Trading |
| `seed_runs/<model>__seed<N>/` | The six training runs — CryptoMamba-v (`cmamba_v`) and CryptoMamba-T (`s5_full`), seeds 23, 24, 25: best-validation checkpoint, `val_preds.csv`, `test_preds.csv`, `done.json`. Seed 23 is the checkpoint the pipeline and the console use | `thesis_pipeline`, `assemble_reproduced_predictions.py`, `build_thesis_artifacts.py`, console |
| `thesis_final/` | Output of `build_thesis_artifacts.py`; untracked, sealed into `../../evidence/` by `package_thesis_evidence.py` | |

The two seed-23 checkpoint paths are pinned by SHA-256 in `../../evidence/model/checkpoint_provenance.json`.

## Environment

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip              # editable install needs pip >= 21.3
./.venv/bin/pip install -e .                                 # CPU, any OS
./.venv/bin/pip install -e ".[gpu]"                          # + native Mamba kernels (Linux + CUDA)
```

`mamba_ssm` and `causal_conv1d` need nvcc and are the optional `gpu` extra. `models/cmamba.py` imports them inside
`try/except` and routes on `tensor.is_cuda`, so a CPU tensor takes the pure-PyTorch `selective_scan_ref` path.

| Task | Needs |
|---|---|
| Evaluation, trading backtest, thesis pipeline | base install, CPU |
| Frozen-checkpoint inference (console Predict) | base install, CPU |
| Training from scratch | `.[gpu]`, Linux + CUDA |

## Commands

```bash
# forecast metrics from a frozen checkpoint
python scripts/evaluation.py --ckpt_path checkpoints/cmamba_v.ckpt --accelerator cpu

# chronological trading backtest (replays forecast_predictions.csv, loads no model)
python scripts/run_backtest.py

# 304-date aligned comparison, paired tests and the corrected self-financing replay
python scripts/build_thesis_artifacts.py

# training (GPU)
python scripts/training.py --config cmamba_v
python scripts/training.py --config s5_full
```

`--config` defaults to `cmamba_v` for both `evaluation.py` and `training.py`.
Expected on the test split with the official checkpoint: RMSE 1598.09, MAPE 2.034 %, MAE 1120.66.

Step-by-step procedure and expected values: [`../../docs/reproduce.md`](../../docs/reproduce.md).
