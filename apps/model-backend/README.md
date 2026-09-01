# model-backend — CryptoMamba research core

Reproduction of **CryptoMamba** (arXiv:2501.01010) for next-day BTC-USD Close forecasting,
plus the CMamba-T / "S5-Full" experiment and the corrected chronological trading backtest.

Scope matches the submitted thesis: the only models with **local** results are **CryptoMamba-v**
(reproduced), **CMamba-T / S5-Full**, and **naive persistence**. The paper's other baselines
(LSTM / GRU / iTransformer / S-Mamba / Bi-LSTM) appear in the thesis only as *paper-reported*
aggregates (Table 4.1, transcribed from the paper) and are **not** re-run here, so their model
code and training configs are not included.

Post-thesis exploration (ModernTCN / TiDE / TSMixer / affine calibration / selective prediction /
checkpoint compression / exogenous track-3 / Colab search runners) is also excluded; it is
preserved on the branch `thesis/pre-monorepo-snapshot` of the original BE repo.

## Layout

| Path | What |
|---|---|
| `models/` | `cmamba` (CryptoMamba-v) and `cmamba_t` (CMamba-T / "S5-Full"); `blueprint_blocks` + `revin` are dependencies of `cmamba_t` |
| `pl_modules/` | Lightning wrappers — `base_module` (shared train/val/predict), `cmamba_module`, `cmamba_t_module`, `data_module` |
| `data_utils/` | `data_transforms`, `dataset`; `exogenous` is imported by the data pipeline (feature builder, inert unless an exogenous config is used) |
| `utils/` | `trade` (paper strategy logic), `io_tools` |
| `thesis_pipeline/` | corrected offline engine: `inference`, `evaluation`, `backtest`, `package` — loads frozen checkpoints, does **not** retrain. Produces the RQ2/RQ3 tables. |
| `configs/` | `data_configs/mode_1.yaml` (paper chronological split); `models/CryptoMamba/{v1,v2,t1,t2}.yaml` (CM-v + CMamba-T w14/w60); `training/cmamba_{v,nv}.yaml` |
| `checkpoints/cmamba_v.ckpt` | official CryptoMamba-v checkpoint |
| `output/` | frozen result artifacts — `evaluation/` (forecast/trading CSVs the thesis cites), `thesis_final/` (RQ2/RQ3 corrected bundle), `reproduce_colab_train/` (the from-scratch RQ1 retrain run + its checkpoint), `improve_track_evidence/s5_full/` (the S5-Full seed-23 checkpoint) |
| `data/` | frozen paper OHLCV cache (`2018-09-17_2024-09-16_86400/`), `paper_reference/`, `one_day_pred.csv` |
| `tests/` | offline tests — backtest, thesis pipeline (`test_thesis_*`), artifact builder |

The two checkpoints the thesis pipeline and the console load are pinned by SHA-256 in
`output/thesis_final/checkpoint_provenance.json`: the reproduced CM-v
(`output/reproduce_colab_train/checkpoints/cmamba_v_best_colab_train.ckpt`) and S5-Full
(`output/improve_track_evidence/s5_full/checkpoints/s5_full__seed23__epoch321-*.ckpt`).

## Environment

`models/cmamba*.py` need native `mamba_ssm` + `causal_conv1d`, which build only on Linux + CUDA
(Google Colab T4/L4). The offline parts (`thesis_pipeline/backtest`, `run_backtest`) run on plain CPU.

```bash
python -m venv .venv && ./.venv/bin/pip install -e .
```

## Reproduce (frozen-checkpoint path — no training)

```bash
python scripts/evaluation.py   --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt
python scripts/run_backtest.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --split test
python -m thesis_pipeline.evaluation     # 304-date controlled comparison: CM-v / S5-Full / naive
python -m thesis_pipeline.backtest       # paper replay + corrected self-financing
```

Outputs land in `output/`. Expected (350-day paper protocol, test): RMSE ≈ 1612.35,
MAPE ≈ 2.05 %, directional accuracy ≈ 56.86 %. Checked-in reference values + SHA-256 sums are in
the evidence submodule (`../../evidence/`).

Upstream research README: `README.upstream.md`.
