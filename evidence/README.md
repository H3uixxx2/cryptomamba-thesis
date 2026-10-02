# Evidence bundle

The checksum-verified result surface the thesis tables and the console read from. The core files
(`forecast/`, `trading/`, `model/`) hold the three models with per-date local results — reproduced
CryptoMamba-v, CryptoMamba-T and naive persistence — for the seed-23 checkpoints. `runs/` holds the
per-seed and ablation tables for all training runs.

```bash
shasum -c SHA256SUMS      # 19 files
```

The console verifies the bundle on startup and reports `NOT_READY` on any mismatch.

## Contents

| File | What it is |
|---|---|
| `forecast/controlled_predictions.csv` | Per-date predictions on the 304 validation and 304 test dates the three models share |
| `forecast/controlled_forecast_metrics.csv` | RMSE, MAE, MAPE, directional accuracy, coverage, parameter count, date range, checkpoint hash |
| `forecast/paired_significance_tests.csv` | Diebold–Mariano on squared error (HAC lag 1) and Wilcoxon signed-rank on absolute error |
| `forecast/paper_reported_metrics.csv` | Aggregate values transcribed from CryptoMamba, Table 3. Labelled `Paper-reported`. |
| `trading/paper_replay_metrics.csv` | Unchanged reproduction of the published strategy replay |
| `trading/corrected_trading_metrics.csv` · `corrected_trading_metadata.json` | Separate same-close self-financing evaluation: fee-aware sizing, reconciled equity |
| `model/s5_full_summary.json` | Run record of the CryptoMamba-T seed-23 checkpoint (`s5_full` is its internal id) |
| `model/checkpoint_provenance.json` | Both seed-23 checkpoints by path, byte count and SHA-256 |
| `runs/controlled_metrics.csv` | Per-run forecast metrics on the 304 shared dates for seeds 23, 24, 25, with family mean and min / max |
| `runs/family_comparisons.csv` | Seed-averaged RMSE differences with block-bootstrap intervals and the reading rule's verdict |
| `runs/direction_intervals.csv` | Directional accuracy by sign agreement on every date, per run, with block-bootstrap intervals |
| `runs/paper_replay.csv` | The paper's strategies on the 350-date validation and test splits, per run |
| `runs/trading_all.csv` | Corrected self-financing results per run: fees 0–0.5 %, next-open fills, borrow cost, validation, test and the period after 09/2024 |
| `runs/ablation_results.json` · `ablation_trading.csv` | The six ablation comparisons and every ablation run |
| `provenance/source_artifact_manifest.json` | Source-to-output hashes from the deterministic build |
| `ARTIFACT_MAP.json` | Which file backs which thesis table and which console screen |

## Notes

- The `paper_reported` rows have no per-date series and never enter a paired test.
- The 304-date sets differ from the 350-date reproduction protocol in step 1 of `docs/reproduce.md`
  and are not pooled with it.
- `trading/paper_replay_metrics.csv` (zero-fee published replay) and `trading/corrected_trading_*`
  (fee-aware self-financing) use different accounting; their balances are not comparable.
- Checkpoint binaries are not stored here; `model/checkpoint_provenance.json` records their paths,
  sizes and SHA-256 values under `apps/model-backend/output/seed_runs/`.
- `forecast/`, `trading/` and `model/` are recomputed from `output/seed_runs/` by the two commands
  below. The `runs/` tables are inputs copied from `thesis_pipeline/data/runs/`; they derive from all
  22 training runs and the period after 09/2024, which are not stored here, so they are not recomputed.
- The per-day corrected equity series is not sealed; its drawdowns and reconciliation errors are
  columns of `trading/corrected_trading_metrics.csv`.

Regenerate with `scripts/build_thesis_artifacts.py` followed by
`scripts/package_thesis_evidence.py` from `apps/model-backend/`.
