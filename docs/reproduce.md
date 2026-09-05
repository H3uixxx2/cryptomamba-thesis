# Reproduce

All commands run from `apps/model-backend/`.

## 0. Environment

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip   # required: see note
./.venv/bin/pip install -e .          # torch, lightning, pandas, scipy — CPU, any OS
./.venv/bin/pip install -e ".[gpu]"   # + mamba-ssm / causal-conv1d — Linux + CUDA, needed to TRAIN
```

The `pip` upgrade is not optional on a fresh venv. This project has a `pyproject.toml` and no
`setup.py`, so an editable install goes through PEP 660, which pip only supports from 21.3.
A stock macOS `python3` (3.9) seeds venvs with pip 21.2.4 and fails with
`Directory cannot be installed in editable mode`. Upgrading pip inside the venv resolves it;
nothing about the package changes.

`mamba_ssm` and `causal_conv1d` require nvcc, which is why they are an extra rather than a base
dependency. Without them the model still runs: `models/cmamba.py` imports the kernels inside
`try/except` and routes on `tensor.is_cuda`, so a CPU tensor takes the pure-PyTorch
`selective_scan_ref` path.

| Step below | Environment |
|---|---|
| 1 – 4 | plain CPU, any OS |
| 5 (training) | Linux + CUDA (Colab T4/L4) |

## 1. Forecast reproduction

```bash
python scripts/evaluation.py --ckpt_path checkpoints/cmamba_v.ckpt --accelerator cpu
```

Writes `output/evaluation/forecast_metrics.csv` and `forecast_predictions.csv`.
`--config` defaults to `cmamba_v`.

Test split, 350 dates (2023-10-01 … 2024-09-14):

| metric | official checkpoint | reproduced checkpoint | paper | reproduced vs paper |
|---|---|---|---|---|
| RMSE | 1598.09 | ≈ 1612.35 | 1598.10 | 0.89 % |
| MAE | 1120.66 | ≈ 1132.54 | 1120.70 | 1.06 % |
| MAPE | 2.034 % | ≈ 2.049 % | 2.034 % | 0.72 % |
| directional accuracy | — | ≈ 56.86 % | — | — |

The official checkpoint reproduces the published figures to the third decimal on CPU. The
reproduced checkpoint is a separate training run and lands within ~1 %.

## 2. Controlled comparison and paired tests

```bash
python -m thesis_pipeline.evaluation
python -m thesis_pipeline.backtest
```

The comparison uses the **304 dates the three models have in common** — S5-Full's 60-day window
consumes more history than CM-v's 14-day window, so their date sets differ, and pooling them would
compare different periods. No interpolation and no cross-date metric combination.

Tests applied: Diebold–Mariano on squared error with HAC lag 1, Wilcoxon signed-rank on absolute
error, a moving-block bootstrap at block lengths L = 5, 7 and 14, and an exact McNemar test on
direction.

Result: on those 304 dates no paired interval and no DM statistic establishes an RMSE advantage
for S5-Full over CM-v; naive persistence has the lowest RMSE on both splits; CM-v has the highest
test directional accuracy at 57.57 %.

The paper's LSTM / GRU / iTransformer / S-Mamba rows are aggregates transcribed from the published
paper. They carry no per-date series, so they are served as `paper_reported` and cannot enter any
paired test.

## 3. Chronological trading backtest

```bash
python scripts/run_backtest.py
```

Loads no model. It replays `output/evaluation/forecast_predictions.csv` through the paper's
strategy logic in `utils/trade.py` and writes `trading_metrics.csv`, `trading_equity_curve.csv`
and `regime_metrics.csv` beside it — 2 checkpoints × 2 splits × 4 strategies × 3 transaction costs
in one run.

Options: `--evaluation_dir` (read/write elsewhere), `--risk` (Smart band, percent), `--balance`
(starting balance).

The decision at day *t* uses only `current_close` and `predicted_close`; `target_close` is used
solely to value the resulting position. At 0 % transaction cost the engine reconciles to the
paper's replay within $0.05.

## 4. Seal the evidence bundle

```bash
python scripts/build_thesis_artifacts.py   # -> output/thesis_final/   (gitignored, regenerable)
python scripts/package_thesis_evidence.py  # -> ../../evidence/ + SHA256SUMS
```

`output/thesis_final/` is intentionally not committed: every file in it is byte-identical to its
counterpart in `evidence/`, and the sealed copy is the one the thesis and the console cite.

Verify what is committed:

```bash
cd ../../evidence && shasum -c SHA256SUMS      # 12 files
```

Steps 2 – 4 are deterministic and reproduce those hashes bit-for-bit. Step 1 depends on the math
library of the machine it runs on and matches to the gaps in the table above.

## 5. Training from scratch (GPU)

```bash
python scripts/training.py --config cmamba_v    # CryptoMamba-v — 14-day window, 136,952 parameters
python scripts/training.py --config s5_full     # CMamba-T / S5-Full — 60-day window, 57,249 parameters
```

`configs/models/CryptoMamba/t2.yaml` builds the released S5-Full graph exactly: 57,249 parameters
with tensor names and shapes identical to the frozen checkpoint, which loads into it with
`load_state_dict`. Reproducing the *weights* additionally needs the original CUDA runtime; the
frozen checkpoints ship so that steps 1 – 4 can be recomputed without training at all.

## Where each number lives

Every table and figure in the thesis is produced here:

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

`evidence/ARTIFACT_MAP.json` maps every thesis table and every evidence-backed console screen to
its source file; `evidence/README.md` states the bundle's scope and boundaries.
