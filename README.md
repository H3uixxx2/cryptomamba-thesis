# CryptoMamba — thesis source code

Reproduction and controlled evaluation of **CryptoMamba** ([arXiv:2501.01010](https://arxiv.org/abs/2501.01010))
for next-day BTC-USD Close forecasting.

Three models have per-date local results: **CryptoMamba-v** (reproduced from scratch),
**CMamba-T / "S5-Full"** (a 60-day-window variant), and **naive persistence**. The paper's other
baselines have no per-date series here — they are carried as published aggregates only.

```bash
git clone https://github.com/H3uixxx2/cryptomamba-thesis.git
```

## Layout

```
cryptomamba-thesis/
├── apps/
│   ├── model-backend/   model, training, evaluation, trading backtest
│   └── console/         5-screen demo: FastAPI backend + React frontend
├── evidence/            checksum-verified result bundle (SHA256SUMS)
└── docs/                architecture.md, reproduce.md
```

| | `apps/model-backend` | `apps/console` |
|---|---|---|
| **Contains** | `models/` `pl_modules/` `data_utils/` `utils/` `thesis_pipeline/` `configs/` `scripts/`, the two frozen checkpoints, the frozen OHLCV split cache, and the result CSVs | `backend/` (FastAPI) + `frontend/` (React) + `web-dist/` (prebuilt bundle) |
| **Stack** | Python 3.9+, PyTorch, Lightning, Mamba SSM | FastAPI, React 19, TypeScript, Tailwind, Plotly |
| **Needs a GPU?** | Only to train. Evaluation, the backtest and frozen-checkpoint inference run on plain CPU, any OS | No |
| **Depends on** | nothing else in this repo | reads `model-backend/output/**` and `evidence/**`; spawns `model-backend/.venv` for Predict-screen inference |

Dependencies point one way. The console never trains and never computes a metric it could not
read from a file; `model-backend` never imports the console.

## Run the demo

```bash
cd apps/console
python3 -m venv backend/.venv && ./backend/.venv/bin/pip install -e backend
./scripts/run_local.sh          # http://127.0.0.1:8600
```

Four screens work from the committed artifacts alone. The Predict screen's frozen-checkpoint
inference additionally needs the `model-backend` environment below — CPU is enough. Without it
the screen surfaces the worker's error instead of a number.

## Reproduce the numbers

```bash
cd apps/model-backend
python -m venv .venv && ./.venv/bin/pip install -e .
python scripts/evaluation.py --ckpt_path checkpoints/cmamba_v.ckpt --accelerator cpu
python scripts/run_backtest.py
```

Test split, official checkpoint: **RMSE 1598.09 · MAPE 2.034 % · MAE 1120.66**.
The reproduced checkpoint lands at RMSE ≈ 1612.35 · MAPE ≈ 2.049 % · directional accuracy ≈ 56.86 %.

Full procedure, including the 304-date paired comparison and training from scratch:
[`docs/reproduce.md`](docs/reproduce.md). Wiring between the two apps:
[`docs/architecture.md`](docs/architecture.md).

## Evidence bundle

`evidence/` holds the numbers the thesis tables cite, each covered by a SHA-256:

```bash
cd evidence && shasum -c SHA256SUMS      # 13 files
```

`ARTIFACT_MAP.json` maps each table and each console screen to the file behind it.
`model/checkpoint_provenance.json` pins both checkpoints by path, byte count and SHA-256.
Every screen fails closed: an unverifiable file yields `NOT_READY` with the reason, never a
substituted value.

## Scope

Included: the code, the frozen inputs needed to run it, and the frozen outputs the thesis cites.

Not included:

| | Why |
|---|---|
| The thesis document (LaTeX, PDF) | Submitted separately. This is a code repository; `docs/reproduce.md` states the expected numbers directly. |
| Model code for LSTM / GRU / iTransformer / S-Mamba | Not re-run for the thesis — their rows are published aggregates transcribed from the paper, with no per-date series, so they cannot enter a paired test. |
| Post-thesis exploration (ModernTCN, TiDE, TSMixer, affine calibration, selective prediction, checkpoint compression, exogenous features, Colab search runners) | Not reported by the thesis. Preserved on branch `thesis/pre-monorepo-snapshot` of the original backend repo. |
| Captures of individual training runs | They duplicated the artifacts that remain and carried files from the retired five-model comparison, so it was ambiguous which copy was authoritative. |
| Virtualenvs, `node_modules`, build caches | Rebuilt from `pyproject.toml` / `pnpm-lock.yaml`. |

## Provenance

| Path | Source |
|---|---|
| `apps/model-backend` | `Crypto-Mamba-BE` @ `thesis/pre-monorepo-snapshot` |
| `apps/console` | `Crypto-Mamba-Console` @ `thesis/final-console-snapshot` |
| `apps/console/backend/src/console_api/vendor/cryptomamba_ui` | `Crypto-Mamba-FE` — 7 modules, verbatim copy, only intra-package imports rewritten |
| `evidence/` | the `final/` bundle of `cryptomamba-thesis-evidence`, flattened in |
