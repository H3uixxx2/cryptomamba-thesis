# model-backend — CryptoMamba research core

The model, its training entry point, the offline evaluation/trading engine, and the frozen
artifacts they produce. No web layer, no live service.

## Models

| Config | Model | Window | Parameters | Target |
|---|---|---|---|---|
| `cmamba_v` → `CMamba_v2` → `configs/models/CryptoMamba/v2.yaml` | CryptoMamba-v (`models/cmamba.py`) | 14 days | 136,952 | next-day Close, raw price |
| `s5_full` → `CMambaT_w60` → `configs/models/CryptoMamba/t2.yaml` | CMamba-T / "S5-Full" (`models/cmamba_t.py`) | 60 days | 57,249 | next-day Close via relative return |

Both are wrapped by `pl_modules/base_module.py`, which owns the shared train/validation/test
steps, the price↔return reconstruction, and denormalisation. `configs/models/archs.yaml` maps the
architecture name in a training config to its model YAML; `utils/io_tools.py` instantiates the
class named in that YAML's `target:` field, so the Lightning wrappers are reached by string
import, not by a static one.

`CMambaT` differs from `CMamba` in one axis: the selective scan runs along the day tokens rather
than the six feature tokens, which decouples window length from model width.

## Layout

| Path | What |
|---|---|
| `models/` | `cmamba.py` (CryptoMamba-v + the pure-PyTorch `selective_scan_ref` CPU fallback), `cmamba_t.py` |
| `pl_modules/` | `base_module.py` (shared steps), `cmamba_module.py`, `cmamba_t_module.py`, `data_module.py` |
| `data_utils/` | `data_transforms.py` (feature tensor + hygiene flags), `dataset.py` (split cache reader) |
| `utils/` | `trade.py` (the paper's strategy logic, unchanged), `io_tools.py` |
| `thesis_pipeline/` | offline engine over frozen checkpoints: `contracts`, `inference`, `evaluation`, `backtest`, `package`. Never trains. |
| `scripts/` | `training.py`, `evaluation.py`, `run_backtest.py`, `checkpoint_inference.py` (stdin/stdout JSON worker), `build_thesis_artifacts.py`, `package_thesis_evidence.py` |
| `configs/` | `data_configs/mode_1.yaml` (chronological split), `models/CryptoMamba/{v2,t2}.yaml`, `models/archs.yaml`, `training/{cmamba_v,s5_full}.yaml` |
| `checkpoints/cmamba_v.ckpt` | the official released checkpoint |
| `data/2018-09-17_2024-09-16_86400/` | the frozen OHLCV split cache: 1461 / 365 / 365 rows |
| `output/` | frozen results — see below |
| `tests/` | 75 offline tests; `pytest tests` |

### Data

No raw source file ships here. `data_utils/dataset.py::get_data` returns the committed
`train.csv` / `val.csv` / `test.csv` directly and only falls back to `data_path` when that cache is
absent, so the cache **is** the dataset. Splits are chronological:

| Split | Range | Rows |
|---|---|---|
| train | 2018-09-17 → 2022-09-16 | 1461 |
| validation | 2022-09-17 → 2023-09-16 | 365 |
| test | 2023-09-17 → 2024-09-15 | 365 |

## What is in `output/`

| Path | Contents | Read by |
|---|---|---|
| `evaluation/forecast_predictions.csv` · `forecast_metrics.csv` | Per-date CM-v predictions and metrics, for the official and the reproduced checkpoint | `thesis_pipeline`, `run_backtest.py`, console Evaluation |
| `evaluation/trading_metrics.csv` · `trading_equity_curve.csv` · `regime_metrics.csv` · `trading_backtest_metadata.json` · `trading_replay_metrics.csv` | Chronological backtest and paper replay | `run_backtest.py`, console Trading |
| `evaluation/offline_prediction.json` | One prediction captured at freeze time; the Predict screen's offline path | console Predict |
| `evaluation/data_quality.csv` | Row counts, date ranges, duplicate/null/monotonicity checks per split | provenance |
| `evaluation/model_selection.json` | Selected checkpoint: epoch, SHA-256, pass status — the record that selection happened before the test split was read | provenance |
| `reproduce_colab_train/checkpoints/cmamba_v_best_colab_train.ckpt` | The reproduced CM-v checkpoint | `thesis_pipeline`, console |
| `improve_track_evidence/s5_full/` | S5-Full: `checkpoints/`, `preds/s5_full__seed23__{val,test}.csv`, `s5_full_summary.json` | `thesis_pipeline`, `build_thesis_artifacts.py`, console |

Both checkpoint paths are pinned by SHA-256 in `../../evidence/model/checkpoint_provenance.json`,
which is itself covered by the bundle's `SHA256SUMS` — so neither file can be moved, renamed or
swapped without verification failing.

`output/thesis_final/` is gitignored: `build_thesis_artifacts.py` regenerates it and
`package_thesis_evidence.py` seals it into `../../evidence/`, where every file is byte-identical.
The sealed copy is the committed one.

## Environment

```bash
python -m venv .venv && ./.venv/bin/pip install -e .        # CPU — any OS, macOS included
./.venv/bin/pip install -e ".[gpu]"                          # + native Mamba kernels (Linux + CUDA)
./.venv/bin/pip install -e ".[dev]"                          # + pytest
```

`mamba_ssm` and `causal_conv1d` need nvcc, so they are the optional `gpu` extra rather than a base
dependency. `models/cmamba.py` imports them inside `try/except` and routes on `tensor.is_cuda`, so
a CPU tensor takes the pure-PyTorch `selective_scan_ref` path whether or not the kernels installed.

| Task | Needs |
|---|---|
| Evaluation, trading backtest, thesis pipeline, tests | base install, CPU |
| Frozen-checkpoint inference (console Predict) | base install, CPU |
| Training from scratch | `.[gpu]`, Linux + CUDA |

## Commands

```bash
# forecast metrics from a frozen checkpoint (no training)
python scripts/evaluation.py --ckpt_path checkpoints/cmamba_v.ckpt --accelerator cpu

# chronological trading backtest — replays forecast_predictions.csv, no model loaded
python scripts/run_backtest.py

# 304-date aligned comparison + paired tests, and the corrected self-financing replay
python -m thesis_pipeline.evaluation
python -m thesis_pipeline.backtest

# training (GPU)
python scripts/training.py --config cmamba_v
python scripts/training.py --config s5_full
```

`--config` defaults to `cmamba_v` for both `evaluation.py` and `training.py`.
Expected on the test split with the official checkpoint: RMSE 1598.09, MAPE 2.034 %, MAE 1120.66.

Step-by-step procedure and the full expected-value table:
[`../../docs/reproduce.md`](../../docs/reproduce.md).
