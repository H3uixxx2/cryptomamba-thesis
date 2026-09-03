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
| `models/` | `cmamba` (CryptoMamba-v) and `cmamba_t` (CMamba-T / "S5-Full") |
| `pl_modules/` | Lightning wrappers — `base_module` (shared train/val/predict), `cmamba_module`, `cmamba_t_module`, `data_module` |
| `data_utils/` | `data_transforms` (feature tensor + the paper's hygiene flags), `dataset` |
| `utils/` | `trade` (paper strategy logic), `io_tools` |
| `thesis_pipeline/` | corrected offline engine: `inference`, `evaluation`, `backtest`, `package` — loads frozen checkpoints, does **not** retrain. Produces the RQ2/RQ3 tables. |
| `configs/` | `data_configs/mode_1.yaml` (paper chronological split); `models/CryptoMamba/v2.yaml` (CM-v) and `t2.yaml` (CMamba-T / S5-Full, 60-day); `training/cmamba_v.yaml` and `training/s5_full.yaml` — the two recipes the thesis reports |
| `checkpoints/cmamba_v.ckpt` | official CryptoMamba-v checkpoint |
| `output/` | frozen result artifacts — `evaluation/` (forecast/trading CSVs the thesis cites), `reproduce_colab_train/checkpoints/` (the reproduced CM-v checkpoint), `improve_track_evidence/s5_full/` (the S5-Full checkpoint + predictions + summary) |
| `data/` | the frozen paper OHLCV split cache `2018-09-17_2024-09-16_86400/` (1461 / 365 / 365 rows). No raw source file ships here: this cache **is** the data authority and `data_utils/dataset.py` reads it directly. |
| `tests/` | offline tests — backtest, thesis pipeline (`test_thesis_*`), artifact builder |

The two checkpoints the thesis pipeline and the console load are pinned by SHA-256 in
`../../evidence/model/checkpoint_provenance.json` (itself covered by the bundle's `SHA256SUMS`):
the reproduced CM-v
(`output/reproduce_colab_train/checkpoints/cmamba_v_best_colab_train.ckpt`) and S5-Full
(`output/improve_track_evidence/s5_full/checkpoints/s5_full__seed23__epoch321-*.ckpt`).

## What is in `output/`

Only frozen results the thesis cites, plus the two checkpoints. No scratch space, and no capture of
any training run — those were removed on purpose: they duplicated these files and still carried
artifacts from the five-model baseline comparison the thesis no longer reports, which made it
ambiguous which copy was authoritative.

| Path | What it is | Who reads it |
|---|---|---|
| `evaluation/forecast_predictions.csv` · `forecast_metrics.csv` | Per-date CM-v predictions and their metrics, official and reproduced checkpoint. **RQ1 evidence.** | `thesis_pipeline`, the console Evaluation screen |
| `evaluation/trading_metrics.csv` · `trading_equity_curve.csv` · `regime_metrics.csv` · `trading_backtest_metadata.json` · `trading_replay_metrics.csv` | The chronological backtest and the paper replay. **RQ3 evidence.** | `scripts/run_backtest.py`, the console Trading screen |
| `evaluation/offline_prediction.json` | One frozen prediction used as the Predict screen's offline backup | console Predict |
| `evaluation/data_quality.csv` | Row counts, date ranges, duplicate/null/monotonicity per split — the record behind the data-scope table | provenance |
| `evaluation/model_selection.json` | Which checkpoint was selected: epoch, SHA-256, PASS. This is what shows the checkpoint was **not** picked after looking at the test split. | provenance |
| `reproduce_colab_train/checkpoints/cmamba_v_best_colab_train.ckpt` | The reproduced CM-v checkpoint (RQ1). Its path and SHA-256 are pinned in the evidence bundle, so it cannot be moved or renamed. | `thesis_pipeline`, the console |
| `improve_track_evidence/s5_full/` | The S5-Full run (RQ4): checkpoint, `preds/s5_full__seed23__{val,test}.csv`, `s5_full_summary.json`. `thesis_pipeline/evaluation.py` reads the two prediction files for the 304-date alignment. | `thesis_pipeline`, `build_thesis_artifacts`, the console |

`output/thesis_final/` is **not** committed: it is the regenerable intermediate that
`scripts/build_thesis_artifacts.py` writes and `scripts/package_thesis_evidence.py` then seals into
`../../evidence/`. Every one of its files is byte-identical to its counterpart there, so the
committed copy is the sealed one.

Everything else that used to live under `output/` — the ModernTCN / TiDE / TSMixer rounds, the
affine-calibration experiment, the selective-prediction and exogenous tracks — was removed: the
thesis does not report any of it. It is preserved on the `thesis/pre-monorepo-snapshot` branch of
the original backend repo.

## What was stripped out of the model code

The exploration rounds that ran after the thesis was frozen left machinery behind in
`base_module.py` and `cmamba_t.py` — a distributional/Student-t/mixture head, a selective
prediction term, a direction auxiliary loss, a `scaled_log_return` target mode, RevIN,
bidirectional scan, recency pooling, and exogenous track-3 feature merging. Every one of
them defaulted to off and **no config, script or pipeline in this repo switched any of them
on**, so they were removed rather than shipped as unreachable branches.

`base_module.py` went from 694 to 209 lines and from 27 to 11 constructor arguments;
`cmamba_t.py` from 70 to 40 lines; `models/revin.py`, `models/blueprint_blocks.py` and
`data_utils/exogenous.py` are gone (the last one also pointed at a `data/exogenous/`
snapshot that does not exist here).

`trade_pnl` and its `madl_temp` argument were **kept**: `validation_step` logs `val/neg_pnl`
for every run, including the thesis path.

How this was checked without a GPU: neither frozen checkpoint stores `hyper_parameters`, so
the constructor signature is not part of the checkpoint contract; the S5-Full `state_dict`
holds only `embed / blocks / norm / head[1,32]` (57,249 parameters, no RevIN, pool or
bi-scan tensors), which is exactly what the trimmed `CMambaT` builds; and a behavioural
harness that instantiates `BaseModule` on both thesis modes (`default` and `ret`) and runs
forward / training_step / validation_step / test_step / denormalize / configure_optimizers
against fixed seeded tensors returns **bit-identical** values before and after the edit.
Full training still runs only on Colab and was not re-executed.

## Environment

```bash
python -m venv .venv && ./.venv/bin/pip install -e .        # CPU: any OS, incl. macOS
./.venv/bin/pip install -e ".[gpu]"                          # + native Mamba kernels (Colab only)
```

The base install carries torch/lightning/pandas/scipy and works everywhere. The native
`mamba_ssm` + `causal_conv1d` kernels build only on Linux + CUDA, so they sit in the optional
`gpu` extra — installing them is required to **train**, not to read the results.

| Task | Needs |
|---|---|
| Offline backtest, thesis pipeline, significance tests | base install, plain CPU |
| Frozen-checkpoint inference (the console Predict screen) | base install, plain CPU — `models/cmamba.py` falls back to the pure-PyTorch `selective_scan_ref` when the kernels are absent |
| Training from scratch | `.[gpu]` on Colab T4/L4 |

## Reproduce (frozen-checkpoint path — no training)

```bash
python scripts/evaluation.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --accelerator cpu
python scripts/run_backtest.py     # reads output/evaluation/forecast_predictions.csv, writes the trading CSVs
python -m thesis_pipeline.evaluation     # 304-date controlled comparison: CM-v / S5-Full / naive
python -m thesis_pipeline.backtest       # paper replay + corrected self-financing
```

Outputs land in `output/`. Expected (350-day paper protocol, test): RMSE ≈ 1612.35,
MAPE ≈ 2.05 %, directional accuracy ≈ 56.86 %. Checked-in reference values + SHA-256 sums are in
the `../../evidence/` bundle (self-verifying via its `SHA256SUMS`).
