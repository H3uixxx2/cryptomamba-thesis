# model-backend — CryptoMamba research core

Reproduction of **CryptoMamba** (arXiv:2501.01010) for next-day BTC-USD Close forecasting,
plus the corrected chronological trading backtest and baseline comparison used in the thesis.

This is the **thesis-core subset** of the original research repo — flat layout kept verbatim so
the code matches what the submitted thesis describes. Post-thesis exploration
(ModernTCN / TiDE / TSMixer / affine calibration / selective prediction / checkpoint
compression / exogenous track-3 / Colab search runners) is **not** included here; it is
preserved on the branch `thesis/pre-monorepo-snapshot` of the original BE repo.

## Layout

| Path | What |
|---|---|
| `models/` | `cmamba` (CryptoMamba-v), `cmamba_t` (CMamba-T / "S5-Full"), `smamba`, `lstm`, `gru`, `iTransformer`, `layers/`; `blueprint_blocks` + `revin` are dependencies of `cmamba_t` |
| `pl_modules/` | Lightning wrappers — `base_module` (shared train/val/predict), per-model modules, `data_module` |
| `data_utils/` | `data_transforms`, `dataset`; `exogenous` is imported by the data pipeline (track-3 feature builder, inert unless a track-3 config is used) |
| `utils/` | `trade` (paper strategy logic), `io_tools` |
| `thesis_pipeline/` | corrected offline engine: `inference`, `evaluation`, `backtest`, `package` (loads frozen checkpoints, does **not** retrain) |
| `configs/` | `data_configs/mode_1.yaml` (paper chronological split), `models/`, `training/` (thesis models only) |
| `scripts/` | CLI entrypoints — `training`, `evaluation`, `evaluate_baselines`, `significance_tests`, `run_backtest`, `simulate_trade`, `one_day_pred`, `merge_baselines`, `export_baseline_predictions`, `build_thesis_artifacts`, `package_thesis_evidence` |
| `checkpoints/cmamba_v.ckpt` | official CryptoMamba-v checkpoint (SHA-256 in `output/thesis_final/checkpoint_provenance.json`) |
| `data/` | frozen paper OHLCV cache (`2018-09-17_2024-09-16_86400/`), `paper_reference/`, `one_day_pred.csv` |
| `output/` | frozen evaluation outputs (`evaluation/`) and the thesis-final bundle (`thesis_final/`) |
| `tests/` | offline tests (backtest / thesis pipeline / baselines) |

## Environment

`models/cmamba*.py` need native `mamba_ssm` + `causal_conv1d`, which build only on Linux + CUDA
(Google Colab T4/L4). The offline parts (`thesis_pipeline/backtest`, `run_backtest`,
`significance_tests`) run on plain CPU.

```bash
python -m venv .venv && ./.venv/bin/pip install -e .
```

## Reproduce (frozen-checkpoint path — no training)

```bash
# forecast + trading evidence from the frozen checkpoint
python scripts/evaluation.py     --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt
python scripts/run_backtest.py   --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --split test
```

Outputs land in `output/`. Expected numbers (test split): RMSE ≈ 1612.35, MAPE ≈ 2.05%,
directional accuracy ≈ 56.86%. See the thesis PDF (`../../thesis/final/draft_offical.pdf`) and
the evidence submodule (`../../evidence/`) for the checked-in reference values + SHA-256 sums.

Upstream research README: `README.upstream.md`.
