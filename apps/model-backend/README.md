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
| `configs/` | `data_configs/mode_1.yaml` (paper chronological split); `models/CryptoMamba/{v1,v2,t1,t2}.yaml` (CM-v + CMamba-T w14/w60); `training/cmamba_{v,nv}.yaml` |
| `checkpoints/cmamba_v.ckpt` | official CryptoMamba-v checkpoint |
| `output/` | frozen result artifacts — `evaluation/` (forecast/trading CSVs the thesis cites), `thesis_final/` (RQ2/RQ3 corrected bundle), `reproduce_colab_train/` (the from-scratch RQ1 retrain run, see below), `improve_track_evidence/s5_full/` (the S5-Full seed-23 checkpoint) |
| `data/` | frozen paper OHLCV cache (`2018-09-17_2024-09-16_86400/`) + `one_day_pred.csv` (the one-day CLI fixture) |
| `tests/` | offline tests — backtest, thesis pipeline (`test_thesis_*`), artifact builder |

The two checkpoints the thesis pipeline and the console load are pinned by SHA-256 in
`output/thesis_final/checkpoint_provenance.json`: the reproduced CM-v
(`output/reproduce_colab_train/checkpoints/cmamba_v_best_colab_train.ckpt`) and S5-Full
(`output/improve_track_evidence/s5_full/checkpoints/s5_full__seed23__epoch321-*.ckpt`).

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

## `output/reproduce_colab_train/` — why it looks duplicated

Two views of the same 2026-06-13 Colab training run, both kept deliberately:

- `runs/cmamba_v_e2e_20260613_085150_UTC/` is the **immutable run bundle exactly as produced**,
  and it is self-verifying: all 30 entries of `provenance/checksums.json` resolve and match
  inside that directory (against the parent directory only 18/30 resolve). It therefore still
  contains artifacts later dropped from the thesis scope — `output/evaluation/baseline_*`, from
  the earlier five-model comparison. Deleting them would break the manifest, so they stay as
  part of the historical record.
- The files beside it — `checkpoints/cmamba_v_best_colab_train.ckpt`, `provenance/`,
  `phase3_inference_bundle.zip` — are the **extraction the rest of the project consumes**. That
  checkpoint path is pinned by SHA-256 in the frozen
  `../../evidence/model/checkpoint_provenance.json`, so it cannot be moved or renamed.

Verify the run bundle:

```bash
cd output/reproduce_colab_train/runs/cmamba_v_e2e_20260613_085150_UTC
python -c "import json,hashlib,pathlib as P;m=json.loads(P.Path('provenance/checksums.json').read_text());print('mismatches:',[k for k,v in m.items() if hashlib.sha256(P.Path(k).read_bytes()).hexdigest()!=v] or 'none',len(m),'entries')"
```

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
the `../../evidence/` bundle (self-verifying via its `SHA256SUMS`).

Upstream research README: `README.upstream.md`.
