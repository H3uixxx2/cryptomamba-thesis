# CryptoMamba — thesis source code

Reproduction and controlled evaluation of **CryptoMamba** (arXiv:2501.01010) for next-day
BTC-USD Close forecasting: the model, the offline evaluation and chronological trading backtest,
the baseline comparison, and a five-screen demo console.

This repository is **code + the evidence it produces** — not the thesis document. See
[*What's not here*](#whats-not-here).

## What's in this repo

Everything here is either code, a frozen input needed to run it, or a frozen output the thesis
cites. Nothing is included "just in case".

| Path | Why it's here |
|---|---|
| `apps/model-backend/` **code** — `models/ pl_modules/ data_utils/ utils/ thesis_pipeline/ configs/ scripts/` | The model, Lightning training, and the offline evaluation + trading engine. This is the research core. |
| `apps/model-backend/checkpoints/cmamba_v.ckpt` | The official checkpoint. Needed to reproduce every forecast/trading number **without** a GPU training run. |
| `apps/model-backend/data/` | The frozen paper OHLCV cache + reference splits. The evaluation and backtest read these directly. |
| `apps/model-backend/output/` | Frozen result artifacts (`evaluation/`, `thesis_final/`, `reproduce_colab_train/`) — the exact numbers the thesis reports, and what the console renders. |
| `apps/console/` | The demo: FastAPI backend (`backend/`) + React frontend (`frontend/`) + a prebuilt bundle (`web-dist/`) so it runs with no Node toolchain. |
| `evidence/` (git submodule) | The checksum-verified bundle: `SHA256SUMS`, both checkpoints, every evaluation CSV, and the `final/` thesis-completion artifacts. This is the canonical, hash-pinned copy of the results. |
| `docs/` | `architecture.md` (how the two apps wire together) and `reproduce.md` (step-by-step). |

## What's not here

| Not included | Where it is | Why not |
|---|---|---|
| **The thesis document** (LaTeX, submitted PDF, Overleaf packages) | Submitted separately through the school | This is a code repository. The committee already has the thesis; shipping a copy of it here only risks version confusion. `docs/reproduce.md` lists the expected numbers directly. |
| **Post-thesis exploration** — ModernTCN / TiDE / TSMixer challengers, affine-calibration and selective-prediction research, checkpoint-compression work, the exogenous track-3 experiment, all Colab search runners, `output/improve_track_evidence/` (132 MB) | Branch `thesis/pre-monorepo-snapshot` of the original backend repo | Not part of the submitted thesis. Kept for audit, out of the delivered scope. |
| Per-app virtualenvs, `node_modules`, build caches | rebuilt locally (`pyproject.toml` / `pnpm-lock.yaml`) | Not source. |

The one exception: `output/evaluation/affine_calibration/predictions.csv` **is** kept — the Predict
screen displays it as a labelled thesis extension — but the calibration *code* is not.

## Layout

```
cryptomamba-thesis/
├── apps/
│   ├── model-backend/   research core — model, training, evaluation, trading backtest
│   └── console/          demo — FastAPI backend + React frontend (Data/Evaluation/Predict/Trading/Architecture)
├── evidence/             git submodule: checksum-verified frozen evidence bundle
└── docs/                 architecture.md, reproduce.md
```

Clone with the evidence submodule:

```bash
git clone --recurse-submodules https://github.com/H3uixxx2/cryptomamba-thesis.git
# after a plain clone:  git submodule update --init
```

## The two apps

| | `apps/model-backend` | `apps/console` |
|---|---|---|
| **What** | CryptoMamba-v + CMamba-T ("S5-Full") + LSTM/GRU/iTransformer baselines; Lightning training; offline evaluation & trading engine (`thesis_pipeline/`) | 5-screen demo: FastAPI (`backend/`) + React (`frontend/`) |
| **Stack** | Python, PyTorch, Lightning, Mamba SSM | FastAPI, React 19, Tailwind, Plotly |
| **Runs on** | Linux + CUDA (Colab) for the model; plain CPU for the offline backtest/eval | any machine (Python 3.9+); the demo bundle needs no Node |
| **Depends on** | nothing else in this repo | reads `model-backend/output/**` + `evidence/final/**`; shells to `model-backend/.venv` for real Predict-screen inference |

The console never trains; it reads what `model-backend` produced. `model-backend` never imports
the console.

## Quick start — run the demo

```bash
cd apps/console
python3 -m venv backend/.venv && ./backend/.venv/bin/pip install -e backend
./scripts/run_local.sh          # http://127.0.0.1:8600
```

Data / Evaluation / Trading / Architecture work from the committed artifacts + evidence bundle.
The Predict screen's *real* frozen-checkpoint inference additionally needs the `model-backend`
environment (below); without it that screen reports `NOT_READY` rather than fabricating a result.

## Reproduce the numbers

See [`docs/reproduce.md`](docs/reproduce.md). Short version, from `apps/model-backend`
(the Mamba model needs the Colab/CUDA env; the backtest step is CPU-only):

```bash
python -m venv .venv && ./.venv/bin/pip install -e .
python scripts/evaluation.py   --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt
python scripts/run_backtest.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --split test
```

Expected (test split): RMSE ≈ 1612.35, MAPE ≈ 2.05 %, directional accuracy ≈ 56.86 %.
Reference values + SHA-256 sums are in the [`evidence/`](evidence) submodule.

## Provenance

- `apps/model-backend` ← `Crypto-Mamba-BE` @ branch `thesis/pre-monorepo-snapshot` (thesis-core subset)
- `apps/console` ← `Crypto-Mamba-Console` @ branch `thesis/final-console-snapshot`
- `apps/console/backend/src/console_api/vendor/cryptomamba_ui` ← `Crypto-Mamba-FE` (7 modules, frozen copy)
- `evidence/` submodule → `cryptomamba-thesis-evidence`
