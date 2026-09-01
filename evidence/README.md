# CryptoMamba-v Final Thesis Evidence

This directory is the checksum-verified evidence surface used by the final
thesis and Console. It contains only the approved local comparison:
reproduced CM-v, CMamba-T/S5-Full, and naive persistence.

## Evidence boundaries

- `forecast/paper_reported_metrics.csv`: aggregate values transcribed from
  CryptoMamba v2, Table 3. These rows are labelled `Paper-reported` and are
  not used in paired tests.
- `forecast/controlled_predictions.csv`: locally aligned per-date predictions
  on 304 validation and 304 test dates per model.
- `forecast/controlled_forecast_metrics.csv`: RMSE, MAE, MAPE, direction,
  coverage, parameters, dates, and checkpoint hashes.
- `forecast/paired_significance_tests.csv`: Diebold-Mariano on squared error
  with HAC lag 1 and Wilcoxon signed-rank on absolute error.
- `trading/paper_replay_metrics.csv`: unchanged reproduction of the published
  strategy replay.
- `trading/corrected_trading_*`: separate same-close self-financing evaluation
  with fee-aware sizing and reconciled equity.
- `model/*`: S5-Full result summary and verified checkpoint provenance.
- `provenance/source_artifact_manifest.json`: source-to-output hashes from the
  deterministic core build.

`SHA256SUMS` covers the 13 regular evidence files. Verify it before consuming
any result. Checkpoint binaries are not duplicated here; their paths, sizes,
and SHA-256 values are recorded in `model/checkpoint_provenance.json`.
