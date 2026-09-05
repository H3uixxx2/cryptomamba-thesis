# CryptoMamba — thesis source code

**This repository is source code, not the thesis.** The thesis document is submitted separately
through the school and is not stored here. What is here is the software that produced every
number that document reports, plus the artifacts those numbers were read from — so the thesis can
be checked against a running program instead of taken on trust.

Subject: reproduction and controlled evaluation of **CryptoMamba**
([arXiv:2501.01010](https://arxiv.org/abs/2501.01010)) for next-day BTC-USD Close forecasting.

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
python3 -m venv backend/.venv
./backend/.venv/bin/python -m pip install --upgrade pip   # see note below
./backend/.venv/bin/pip install -e backend
./scripts/run_local.sh          # http://127.0.0.1:8600
```

Four screens work from the committed artifacts alone. The Predict screen's frozen-checkpoint
inference additionally needs the `model-backend` environment below — CPU is enough. Without it
the screen surfaces the worker's error instead of a number.

## Reproduce the numbers

```bash
cd apps/model-backend
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip           # see note below
./.venv/bin/pip install -e .
./.venv/bin/python scripts/evaluation.py --ckpt_path checkpoints/cmamba_v.ckpt --accelerator cpu
./.venv/bin/python scripts/run_backtest.py
```

> **Upgrade `pip` inside the fresh venv first.** An editable install from a `pyproject.toml`
> with no `setup.py` needs pip ≥ 21.3 (PEP 660). A stock macOS `python3` (3.9) creates venvs
> with pip 21.2.4, which fails with *"Directory cannot be installed in editable mode"*.

Test split, official checkpoint: **RMSE 1598.09 · MAPE 2.034 % · MAE 1120.66**.
The reproduced checkpoint lands at RMSE ≈ 1612.35 · MAPE ≈ 2.049 % · directional accuracy ≈ 56.86 %.

Full procedure, including the 304-date paired comparison and training from scratch:
[`docs/reproduce.md`](docs/reproduce.md). Wiring between the two apps:
[`docs/architecture.md`](docs/architecture.md).

## Evidence bundle

`evidence/` holds the numbers the thesis tables cite, each covered by a SHA-256:

```bash
cd evidence && shasum -c SHA256SUMS      # 12 files
```

`ARTIFACT_MAP.json` maps each table and each console screen to the file behind it.
`model/checkpoint_provenance.json` pins both checkpoints by path, byte count and SHA-256.
Every screen fails closed: an unverifiable file yields `NOT_READY` with the reason, never a
substituted value.

## Scope — what this repository is, and what it covers

It is the executable half of the work: the model, the pipeline that turns a frozen checkpoint into
the reported tables, the demo that renders them, and the checksum-verified artifacts those tables
cite. Every file is either code that runs, an input that code reads, or an output the thesis
reports. There is no thesis text, no LaTeX, no PDF and no submission material here.

Three properties follow, and they are the point of shipping the code at all:

- **Every reported number is recomputable.** Both checkpoints ship, so the forecast metrics, the
  304-date paired comparison, the robustness intervals and the trading tables regenerate from a
  plain clone on CPU — no GPU, no retraining, no network.
- **Every number is traceable to a file.** `evidence/SHA256SUMS` covers all 12 artifacts, and
  `evidence/ARTIFACT_MAP.json` names the artifact behind each thesis table and each evidence-backed
  console screen (Evaluation, Trading, Architecture; Data and Predict read the split cache and the
  checkpoints instead, both pinned in `evidence/model/checkpoint_provenance.json`).
- **Nothing is estimated.** Where an artifact is missing or fails verification, the console reports
  `NOT_READY` with the reason rather than substituting a value.

### Coverage of the thesis

Every table and figure in the submitted document is produced here. Nothing it reports was computed
somewhere else:

| Thesis | Produced by | Read from |
|---|---|---|
| Tables 3.1, 3.2 — splits and model contracts | `configs/data_configs/mode_1.yaml`, `thesis_pipeline/contracts.py`, `data_utils/data_transforms.py` | `data/2018-09-17_2024-09-16_86400/` (1461 / 365 / 365) |
| §4.1 — 350-date RQ1 reproduction | `scripts/evaluation.py` | `output/evaluation/forecast_metrics.csv` |
| Table 4.1 — paper-reported reference | transcribed, never recomputed | `evidence/forecast/paper_reported_metrics.csv` |
| Tables 4.2, 4.3 · Figures 4.1, 5.1 — 304-date controlled comparison | `python -m thesis_pipeline.evaluation` | `evidence/forecast/controlled_forecast_metrics.csv` |
| Tables 4.4, 4.5, 4.6 — block bootstrap, block-length sensitivity, chronological halves | `console_api/loaders/forecast_robustness.py` (50,000 resamples, seed 230813, L = 5/7/14), served by `GET /api/reproduce` | `evidence/forecast/controlled_predictions.csv` |
| Table 4.7 — Diebold–Mariano and Wilcoxon paired tests | `python -m thesis_pipeline.evaluation` | `evidence/forecast/paired_significance_tests.csv` |
| §4.2 — exact McNemar on direction | `console_api/loaders/forecast_robustness.py`, served by `GET /api/reproduce` | `evidence/forecast/controlled_predictions.csv` |
| Table 4.8 — paper trading replay | `scripts/run_backtest.py` | `evidence/trading/paper_replay_metrics.csv` |
| Table 4.9 · Figure 4.2 — corrected self-financing results, fee sensitivity | `python -m thesis_pipeline.backtest` | `evidence/trading/corrected_trading_metrics.csv` |
| Figure 2.1 — released CM-v scan axis vs the paper diagram | `models/cmamba.py` vs `models/cmamba_t.py`; rendered by `frontend/src/components/ScanAxisDiagram.tsx` | source |
| Table 6.1 · Figures 6.1–6.4 — the five Console screens | `apps/console` | the artifacts above |

The document carries the expected values for each of these, so the two can be checked against each
other independently: run the commands above, then compare.

### Boundaries

- Training is the one step not reproducible from a plain clone on CPU: the native Mamba kernels
  need Linux + CUDA. Both frozen checkpoints ship precisely so that everything downstream of
  training does not depend on re-running it.
- The 350-date and 304-date protocols are different target sets and are never pooled; the paper's
  aggregate rows have no per-date series and never enter a paired test.

## Provenance

| Path | Source |
|---|---|
| `apps/model-backend` | `Crypto-Mamba-BE` @ `thesis/pre-monorepo-snapshot` |
| `apps/console` | `Crypto-Mamba-Console` @ `thesis/final-console-snapshot` |
| `apps/console/backend/src/console_api/vendor/cryptomamba_ui` | `Crypto-Mamba-FE` — 7 modules, verbatim copy, only intra-package imports rewritten |
| `evidence/` | the `final/` bundle of `cryptomamba-thesis-evidence`, flattened in |
