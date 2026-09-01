# CryptoMamba — thesis source

Reproduction and controlled evaluation of **CryptoMamba** (arXiv:2501.01010) for next-day
BTC-USD Close forecasting, plus a corrected chronological trading backtest, baseline comparison,
and a five-screen demo console.

This repository is the complete source for the thesis. The submitted PDF is in
[`thesis/final/draft_offical.pdf`](thesis/final/draft_offical.pdf).

## Layout

```
cryptomamba-thesis/
├── apps/
│   ├── model-backend/   research core — model, training, evaluation, trading backtest
│   └── console/          demo — FastAPI backend + React frontend (Data/Evaluation/Predict/Trading/Architecture)
├── evidence/             git submodule: checksum-verified frozen evidence bundle
├── thesis/               LaTeX sources (VI + EN), submitted PDF, Overleaf packages
└── docs/                 architecture.md, reproduce.md
```

Clone with the evidence submodule:

```bash
git clone --recurse-submodules <repo-url>
# or, after a plain clone:
git submodule update --init
```

## The two apps

| | `apps/model-backend` | `apps/console` |
|---|---|---|
| **What** | CryptoMamba-v + CMamba-T ("S5-Full") + LSTM/GRU/iTransformer baselines; Lightning training; offline evaluation & trading engine (`thesis_pipeline/`) | 5-screen demo: FastAPI (`backend/`) + React (`frontend/`) |
| **Stack** | Python, PyTorch, Lightning, Mamba SSM | FastAPI, React 19, Tailwind, Plotly |
| **Runs on** | Linux + CUDA (Colab) for the model; plain CPU for the offline backtest/eval | any machine (Python 3.9+); the demo bundle needs no Node |
| **Depends on** | nothing in this repo | reads `model-backend/output/**` + `evidence/final/**`; shells to `model-backend/.venv` for real Predict-screen inference |

The console never trains; it reads what `model-backend` produced. `model-backend` never imports
the console.

## Quick start — run the demo

```bash
cd apps/console
python3 -m venv backend/.venv && ./backend/.venv/bin/pip install -e backend
./scripts/run_local.sh          # http://127.0.0.1:8600
```

The Data/Evaluation/Trading/Architecture screens work from the committed artifacts + evidence
bundle. The Predict screen's *real* frozen-checkpoint inference additionally needs the
`model-backend` environment (below); without it that screen reports `NOT_READY` rather than
fabricating a result.

## Reproduce the numbers

See [`docs/reproduce.md`](docs/reproduce.md). Short version, from `apps/model-backend`
(needs the Colab/CUDA env for the Mamba model; the backtest step is CPU-only):

```bash
python -m venv .venv && ./.venv/bin/pip install -e .
python scripts/evaluation.py   --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt
python scripts/run_backtest.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --split test
```

Expected (test split): RMSE ≈ 1612.35, MAPE ≈ 2.05%, directional accuracy ≈ 56.86%.
Reference values + SHA-256 sums are in the [`evidence/`](evidence) submodule.

## Deliberately excluded

Post-thesis exploration is **not** in this repo (kept on the branch `thesis/pre-monorepo-snapshot`
of the original backend repo): ModernTCN / TiDE / TSMixer challengers, affine-calibration and
selective-prediction research code, checkpoint-compression work, the exogenous track-3 feature
experiment, and all Colab search runners. The frozen `affine_calibration/` *prediction artifact*
is kept only because the Predict screen displays it as a labelled thesis extension.

## Provenance

- `apps/model-backend` ← `Crypto-Mamba-BE` @ branch `thesis/pre-monorepo-snapshot` (thesis-core subset)
- `apps/console` ← `Crypto-Mamba-Console` @ branch `thesis/final-console-snapshot`
- `apps/console/backend/src/console_api/vendor/cryptomamba_ui` ← `Crypto-Mamba-FE` (7 modules, frozen)
- `evidence/` submodule → `cryptomamba-thesis-evidence`
