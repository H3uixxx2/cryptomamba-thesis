# Evidence bundle

The checksum-verified result surface the thesis tables and the console read from. It holds only
the three models with per-date local results: reproduced CryptoMamba-v, CMamba-T / S5-Full, and
naive persistence.

```bash
shasum -c SHA256SUMS      # 13 files
```

Verify before consuming any value. The console verifies on startup and reports `NOT_READY` on any
mismatch rather than serving unverified data.

## Contents

| File | What it is |
|---|---|
| `forecast/controlled_predictions.csv` | Per-date predictions on the 304 validation and 304 test dates the three models share |
| `forecast/controlled_forecast_metrics.csv` | RMSE, MAE, MAPE, directional accuracy, coverage, parameter count, date range, checkpoint hash |
| `forecast/paired_significance_tests.csv` | Diebold–Mariano on squared error (HAC lag 1) and Wilcoxon signed-rank on absolute error |
| `forecast/paper_reported_metrics.csv` | Aggregate values transcribed from CryptoMamba, Table 3. Labelled `Paper-reported`. |
| `trading/paper_replay_metrics.csv` | Unchanged reproduction of the published strategy replay |
| `trading/corrected_trading_metrics.csv` · `corrected_trading_equity.csv` · `corrected_trading_metadata.json` | Separate same-close self-financing evaluation: fee-aware sizing, reconciled equity |
| `model/s5_full_summary.json` | S5-Full run summary |
| `model/checkpoint_provenance.json` | Both checkpoints by path, byte count and SHA-256 |
| `provenance/source_artifact_manifest.json` | Source-to-output hashes from the deterministic build |
| `ARTIFACT_MAP.json` | Which file backs which thesis table and which console screen |

## Boundaries

- The `paper_reported` rows have no per-date series and therefore never enter a paired test. They
  are a published reference, not a locally reproduced result.
- The 304-date sets differ from the 350-date reproduction protocol in step 1 of
  `docs/reproduce.md`. They are not interchangeable and are never pooled.
- `trading/paper_replay_metrics.csv` and `trading/corrected_trading_*` use different accounting
  rules — zero-fee published replay versus fee-aware self-financing — so their balances are not
  directly comparable.
- Checkpoint binaries are not duplicated here. `model/checkpoint_provenance.json` records their
  paths, sizes and SHA-256 values under `apps/model-backend/output/`.

Regenerate with `scripts/build_thesis_artifacts.py` followed by
`scripts/package_thesis_evidence.py` from `apps/model-backend/`.
