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
python scripts/assemble_reproduced_predictions.py   # reproduced CM-v, seed 23
```

The first command writes `output/evaluation/forecast_metrics.csv` and `forecast_predictions.csv`
for the official checkpoint (`--config` defaults to `cmamba_v`). The second replaces the
`retrained_checkpoint` rows from the seed-23 training run stored in
`output/seed_runs/cmamba_v__seed23/`: it reads that run's stored predictions, checks that their
price columns equal the frozen price base date by date, and recomputes the 350-date test metrics and
the paper's strategy replay. No model is loaded and nothing is estimated. Pass `--run` to assemble
another run.

Test split, 350 dates (2023-10-01 … 2024-09-14):

| metric | official checkpoint | reproduced checkpoint (seed 23) | paper | reproduced vs paper |
|---|---|---|---|---|
| RMSE | 1598.09 | 1604.45 | 1598.10 | 0.40 % |
| MAE | 1120.66 | 1128.00 | 1120.70 | 0.65 % |
| MAPE | 2.034 % | 2.049 % | 2.034 % | 0.72 % |
| directional accuracy (sign agreement) | 55.43 % | 54.86 % | — | — |

The official checkpoint reproduces the published figures to the third decimal on CPU. The
reproduced checkpoint is a separate training run of the fork code on a Tesla T4. Seeds 24 and 25 of
the same recipe give test RMSE 1613.15 and 1611.40, computed from
`output/seed_runs/cmamba_v__seed{24,25}/test_preds.csv` with the same formulas.

## 2. Controlled comparison and paired tests

The comparison is computed by `thesis_pipeline.evaluation`, which `scripts/build_thesis_artifacts.py`
calls in step 4.

It uses the **304 dates the three models have in common** — CryptoMamba-T's 60-day window consumes
more history than CM-v's 14-day window, so their date sets differ, and pooling them would compare
different periods. No interpolation and no cross-date metric combination.

Tests applied: Diebold–Mariano on squared error with HAC lag 1, Wilcoxon signed-rank on absolute
error, a moving-block bootstrap at block lengths L = 5, 7 and 14, and an exact McNemar test on
direction.

Result for the seed-23 checkpoints: on those 304 dates no paired test establishes an RMSE advantage
for CryptoMamba-T over CM-v (Diebold–Mariano p = 0.094 on validation, 0.638 on test); naive
persistence has the lowest RMSE on both splits (554.94 and 1657.55); CM-v has the highest test
directional accuracy (56.58 % against 49.01 % for CryptoMamba-T).

`evidence/runs/` adds what one checkpoint per model cannot show: the three seeds of each model, the
seed-averaged comparisons with their block-bootstrap intervals, the ablation models, and trading with
next-open fills, borrow cost and the period after 09/2024. Those tables come from all 22 training
runs; only the six seed runs above are stored in this repository, so `runs/` is a copied input and is
not recomputed here (see `evidence/README.md`).

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
`build_thesis_artifacts.py` reads the seed-23 runs in `output/seed_runs/`, runs the date-aligned
evaluation and the corrected self-financing backtest, and copies the `runs/` inputs from
`thesis_pipeline/data/runs/`.

Verify what is committed:

```bash
cd ../../evidence && shasum -c SHA256SUMS      # 19 files
```

Steps 2 – 4 are deterministic and reproduce those hashes bit-for-bit. Step 1 depends on the math
library of the machine it runs on and matches to the gaps in the table above.

## 5. Training from scratch (GPU)

```bash
python scripts/training.py --config cmamba_v    # CryptoMamba-v — 14-day window, 136,952 parameters
python scripts/training.py --config s5_full     # CryptoMamba-T — 60-day window, 57,249 parameters
```

`--seed` sets the random seed (the runs in `output/seed_runs/` use 23, 24 and 25). `s5_full` is the
internal name of CryptoMamba-T.

`configs/models/CryptoMamba/t2.yaml` builds the released CryptoMamba-T graph exactly: 57,249
parameters with tensor names and shapes identical to the seed-23 checkpoint, which loads into it with
`load_state_dict`. Reproducing the *weights* additionally needs the original CUDA runtime; the
checkpoints ship so that steps 1 – 4 can be recomputed without training at all.

## Where each number lives

Each number in the thesis traces to a file here. `evidence/runs/` holds copied inputs rather than
recomputed values, as step 2 explains.

| Thesis | Produced by | Read from |
|---|---|---|
| Ch. 3 — splits, model contracts, training set-up | `configs/data_configs/mode_1.yaml`, `configs/training/{cmamba_v,s5_full}.yaml`, `thesis_pipeline/contracts.py`, `data_utils/data_transforms.py` | `data/2018-09-17_2024-09-16_86400/` (1461 / 365 / 365) |
| Ch. 4 — 350-date reproduction (RQ1), official checkpoint and seed 23 | `scripts/evaluation.py`, `scripts/assemble_reproduced_predictions.py` | `output/evaluation/forecast_metrics.csv` |
| Ch. 4 — seeds 24 and 25 | the same formulas on the stored predictions | `output/seed_runs/cmamba_v__seed{24,25}/test_preds.csv` |
| Table 1.1 — paper-reported reference | transcribed, never recomputed | `evidence/forecast/paper_reported_metrics.csv` |
| Ch. 5 — 304-date controlled comparison and paired tests (seed 23) | `thesis_pipeline.evaluation`, via `scripts/build_thesis_artifacts.py` | `evidence/forecast/controlled_forecast_metrics.csv`, `paired_significance_tests.csv` |
| Ch. 5 — three seeds per model, seed-family comparisons, directional intervals | copied inputs | `evidence/runs/controlled_metrics.csv`, `family_comparisons.csv`, `direction_intervals.csv` |
| Block bootstrap and McNemar shown on the Evaluation screen | `console_api/loaders/forecast_robustness.py` (50,000 resamples, seed 230813, L = 5/7/14), served by `GET /api/reproduce` | `evidence/forecast/controlled_predictions.csv` |
| Ch. 6 — paper replay | `output/evaluation/trading_replay_metrics.csv` (official rows unchanged; reproduced rows by `scripts/assemble_reproduced_predictions.py`), copied by `scripts/build_thesis_artifacts.py`; per-run values are copied inputs | `evidence/trading/paper_replay_metrics.csv`, `evidence/runs/paper_replay.csv` |
| Ch. 6 — corrected self-financing, fee sensitivity (seed 23) | `thesis_pipeline.backtest`, via `scripts/build_thesis_artifacts.py` | `evidence/trading/corrected_trading_metrics.csv` |
| Ch. 6, App. B — per-seed trading, fills, borrow cost, the period after 09/2024 | copied inputs | `evidence/runs/trading_all.csv` |
| Ch. 7 — CryptoMamba-T and the ablation | copied inputs | `evidence/runs/ablation_results.json`, `ablation_trading.csv` |
| Ch. 8 — the four Console screens | `apps/console` | the artifacts above |

`evidence/ARTIFACT_MAP.json` maps every evidence-backed console screen to its source file;
`evidence/README.md` states the bundle's scope and boundaries.
