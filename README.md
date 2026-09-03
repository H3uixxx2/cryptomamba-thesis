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

This repository is the executable half of the work: the model, the pipeline that turns a frozen
checkpoint into the reported tables, the demo that renders them, and the checksum-verified
artifacts those tables cite. Every file here is either code that runs, an input that code reads,
or an output the thesis reports.

That gives three properties worth stating up front:

- **Every reported number is recomputable.** Both checkpoints ship, so the forecast metrics, the
  304-date paired comparison and the trading tables can be regenerated from a plain clone on CPU,
  without a GPU and without retraining.
- **Every number is traceable to a file.** `evidence/ARTIFACT_MAP.json` names the artifact behind
  each thesis table and each console screen, and `SHA256SUMS` covers all of them.
- **Nothing is estimated.** Where an artifact is missing or fails verification, the console reports
  `NOT_READY` with the reason rather than substituting a value.

The thesis document itself is submitted through the school; `docs/reproduce.md` carries the
expected values, so the two can be checked against each other independently.

## Provenance

| Path | Source |
|---|---|
| `apps/model-backend` | `Crypto-Mamba-BE` @ `thesis/pre-monorepo-snapshot` |
| `apps/console` | `Crypto-Mamba-Console` @ `thesis/final-console-snapshot` |
| `apps/console/backend/src/console_api/vendor/cryptomamba_ui` | `Crypto-Mamba-FE` — 7 modules, verbatim copy, only intra-package imports rewritten |
| `evidence/` | the `final/` bundle of `cryptomamba-thesis-evidence`, flattened in |
