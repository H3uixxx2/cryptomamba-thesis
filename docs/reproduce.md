# Reproduce

All commands run from `apps/model-backend/`.

## Environments

| Step | Environment |
|---|---|
| Train / evaluate the Mamba model | Linux + CUDA (Google Colab T4/L4). Native `mamba_ssm` + `causal_conv1d` do **not** build on macOS. |
| Offline trading backtest, significance tests, thesis pipeline | plain CPU, any OS |

```bash
python -m venv .venv
./.venv/bin/pip install -e .          # installs from requirements.txt (torch, lightning, mamba-ssm, ...)
```

## 1. Forecast reproduction (frozen official checkpoint)

```bash
python scripts/evaluation.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt
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
                                         # -> output/thesis_final/{controlled_forecast_metrics,paired_significance_tests}.csv
```

Diebold–Mariano (squared error, HAC lag 1) and Wilcoxon (absolute error), plus a
moving-block bootstrap at block lengths L = 5, 7, 14 and an exact McNemar test on direction.
On the 304 common dates, neither paired interval nor DM confirms an RMSE advantage for S5-Full
over CM-v; persistence has the lowest RMSE on both splits; CM-v has the highest test directional
accuracy (57.57 %). The paper's LSTM / GRU / iTransformer / S-Mamba baselines are used only as
published aggregates (`docs/` Table 4.1) and are not recomputed here.

## 3. Chronological trading backtest (CPU-only)

```bash
python scripts/run_backtest.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --split test
```

Writes `output/evaluation/trading_metrics.csv`, `trading_equity_curve.csv`, `regime_metrics.csv`.
At 0 % transaction cost the engine reconciles to the paper's replay within < $0.05.

## 4. Corrected thesis pipeline (frozen checkpoints, no training)

```bash
python -m thesis_pipeline.evaluation      # 304 common-date controlled comparison: CM-v / S5-Full / naive
python -m thesis_pipeline.backtest        # corrected self-financing replay
python -m thesis_pipeline.package         # assemble the final evidence bundle
```

## Checking against the evidence submodule

`evidence/final/SHA256SUMS` lists every frozen artifact with its hash. Regenerated outputs should
match those hashes bit-for-bit for the deterministic steps (2, 3, 4); step 1 depends on the CUDA
math library version and matches to the reported gaps above.

```bash
cd ../../evidence/final && shasum -c SHA256SUMS
```
