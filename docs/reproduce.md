# Reproduce

All commands run from `apps/model-backend/`.

## Environments

| Step | Environment |
|---|---|
| Train the Mamba model | Linux + CUDA (Google Colab T4/L4) — native `mamba_ssm` + `causal_conv1d` do **not** build on macOS, so they are an optional extra |
| Frozen-checkpoint inference, offline trading backtest, significance tests, thesis pipeline | plain CPU, any OS |

```bash
python -m venv .venv
./.venv/bin/pip install -e .          # torch, lightning, pandas, scipy — CPU, any OS
./.venv/bin/pip install -e ".[gpu]"   # + mamba-ssm/causal-conv1d, Colab only, needed to TRAIN
```

Without the `gpu` extra the model still runs on CPU: `models/cmamba.py` imports the CUDA kernels
in a `try/except` and falls back to its pure-PyTorch `selective_scan_ref`.

## 1. Forecast reproduction (frozen official checkpoint)

```bash
python scripts/evaluation.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --accelerator cpu
```

Writes `output/evaluation/forecast_metrics.csv` + `forecast_predictions.csv`.
Expected on the paper test split (2023-10-01 … 2024-09-14):

| metric | value | paper | gap |
|---|---|---|---|
| RMSE | ≈ 1612.35 | 1598.10 | 0.89 % |
| MAE  | ≈ 1132.54 | 1120.70 | 1.06 % |
| MAPE | ≈ 2.049 % | 2.034 % | 0.72 % |
| directional accuracy | ≈ 56.86 % | — | — |

## 2. Controlled comparison + paired tests (304 common dates)

```bash
python -m thesis_pipeline.evaluation     # CM-v / S5-Full / naive persistence on the 304 aligned dates
                                         # -> output/thesis_final/ (regenerated; not committed —
                                         #    the sealed copy lives in evidence/forecast/)
```

Diebold–Mariano (squared error, HAC lag 1) and Wilcoxon (absolute error), plus a
moving-block bootstrap at block lengths L = 5, 7, 14 and an exact McNemar test on direction.
On the 304 common dates, neither paired interval nor DM confirms an RMSE advantage for S5-Full
over CM-v; persistence has the lowest RMSE on both splits; CM-v has the highest test directional
accuracy (57.57 %). The paper's LSTM / GRU / iTransformer / S-Mamba baselines are used only as
published aggregates (`docs/` Table 4.1) and are not recomputed here.

## 3. Chronological trading backtest (CPU-only)

```bash
python scripts/run_backtest.py        # no model needed: it replays forecast_predictions.csv
```

Reads `output/evaluation/forecast_predictions.csv` and writes `trading_metrics.csv`,
`trading_equity_curve.csv`, `regime_metrics.csv` beside it — all 2 checkpoints x 2 splits x
4 strategies x 3 costs in one run. `--evaluation_dir` points it somewhere else; `--risk` and
`--balance` change the Smart band and the starting balance.
At 0 % transaction cost the engine reconciles to the paper's replay within < $0.05.

## 4. Corrected thesis pipeline (frozen checkpoints, no training)

```bash
python -m thesis_pipeline.evaluation      # 304 common-date controlled comparison: CM-v / S5-Full / naive
python -m thesis_pipeline.backtest        # corrected self-financing replay
python scripts/build_thesis_artifacts.py  # -> output/thesis_final/  (regenerable, gitignored)
python scripts/package_thesis_evidence.py # seals thesis_final/ -> ../../evidence/ with SHA256SUMS
```

`output/thesis_final/` is deliberately not committed: every file in it is byte-identical to its
counterpart in `evidence/`, and the sealed copy is the one the thesis and the console cite.

## 5. Training from scratch (GPU / Colab only)

The thesis reports two trained models, and each has its recipe here:

```bash
python scripts/training.py --config cmamba_v    # CryptoMamba-v, 14-day window, 136,952 parameters
python scripts/training.py --config s5_full     # CMamba-T / S5-Full, 60-day window, 57,249 parameters
```

Both need `pip install -e ".[gpu]"` at full budget. `configs/models/CryptoMamba/t2.yaml` builds
exactly the released S5-Full graph — 57,249 parameters, identical tensor names and shapes, and the
frozen checkpoint loads into it with `load_state_dict` — so the architecture in this repo is the one
that produced the RQ4 numbers. Reproducing the *weights* additionally requires the original Colab
CUDA runtime; the frozen checkpoints are provided so every result below can be recomputed without
training.

## Checking against the evidence bundle

`evidence/SHA256SUMS` lists every frozen artifact with its hash. Regenerated outputs should
match those hashes bit-for-bit for the deterministic steps (2, 3, 4); step 1 depends on the CUDA
math library version and matches to the reported gaps above.

```bash
cd ../../evidence && shasum -c SHA256SUMS
```

`evidence/README.md` states the bundle's scope; `evidence/ARTIFACT_MAP.json` maps each thesis
table to its file.
