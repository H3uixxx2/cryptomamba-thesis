# CryptoMamba — thesis source code

Reproduction and controlled evaluation of [CryptoMamba](https://arxiv.org/abs/2501.01010)
(arXiv:2501.01010) for next-day BTC-USD Close forecasting: the model, the offline
evaluation/trading pipeline, a 5-screen demo console, and the checksum-verified result bundle.

## Requirements

| | |
|---|---|
| Python | 3.9+ |
| pip | **≥ 21.3** — upgrade inside each venv; stock macOS seeds 21.2.4, which cannot do a PEP 660 editable install |
| OS | any; evaluation, backtest and inference run on CPU |
| GPU | only to train from scratch (Linux + CUDA, `pip install -e ".[gpu]"`) |
| Node | not required — `apps/console/web-dist/` is committed |

## Install

```bash
# console
python3 -m venv apps/console/backend/.venv
apps/console/backend/.venv/bin/python -m pip install --upgrade pip
apps/console/backend/.venv/bin/pip install -e apps/console/backend

# model
python3 -m venv apps/model-backend/.venv
apps/model-backend/.venv/bin/python -m pip install --upgrade pip
apps/model-backend/.venv/bin/pip install -e apps/model-backend
```

## Usage

Each block is self-contained and starts from the repository root.

```bash
# demo console -> http://127.0.0.1:8600
cd apps/console && ./scripts/run_local.sh
```

```bash
# forecast metrics from the official checkpoint
cd apps/model-backend && ./.venv/bin/python scripts/evaluation.py \
    --ckpt_path checkpoints/cmamba_v.ckpt --accelerator cpu
```

```bash
# chronological trading backtest
cd apps/model-backend && ./.venv/bin/python scripts/run_backtest.py
```

```bash
# verify the sealed artifacts
cd evidence && shasum -c SHA256SUMS
```

| Command | Expected |
|---|---|
| `evaluation.py` | test split — RMSE `1598.09` · MAPE `2.034 %` · MAE `1120.66` |
| `run_backtest.py` | `output/evaluation/{trading_metrics,trading_equity_curve,regime_metrics}.csv` |
| `shasum -c SHA256SUMS` | 12 files, all `OK` |
| `run_local.sh` | Data · Evaluation · Predict · Trading · Architecture |

The console's four other screens read committed artifacts; Predict runs real inference in the
model-backend venv.

## Documentation

| | |
|---|---|
| [`docs/reproduce.md`](docs/reproduce.md) | full procedure, expected values, training |
| [`docs/architecture.md`](docs/architecture.md) | how the two apps fit together, provenance |
| [`apps/model-backend/README.md`](apps/model-backend/README.md) | models, configs, `output/` |
| [`apps/console/README.md`](apps/console/README.md) | API, screens, wiring |
| [`evidence/README.md`](evidence/README.md) | what each sealed artifact is |

The thesis document is submitted separately and is not in this repository.
