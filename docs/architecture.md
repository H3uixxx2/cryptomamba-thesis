# Architecture

Two apps and one evidence bundle. Dependencies point one way — `console → model-backend` and
`console → evidence`; `model-backend` depends on nothing else in the repo.

```
  train / evaluate            ┌───────────────────────────────┐
  (CUDA to train,             │       apps/model-backend      │
   CPU to evaluate)  ───────▶ │  models/  pl_modules/         │
                              │  data_utils/  thesis_pipeline/│
                              │                               │
                              │  writes ──▶ output/evaluation/│──────────┐
                              │             checkpoints       │          │ files
                              └───────────────┬───────────────┘          │ (read-only)
                                              │                          │
                          subprocess: one JSON │                          │
                          request per Predict  │                          │
                          call, bounded        ▼                          ▼
  ┌────────────────────┐               ┌──────────────────────────────────────┐
  │      evidence/     │  verify once  │        apps/console/backend          │
  │  SHA256SUMS        │──────────────▶│  routers/ → services/ → loaders/     │
  │  forecast/ trading/│               │  core/   schemas/   vendor/          │
  │  model/ provenance/│               └──────────────────┬───────────────────┘
  └────────────────────┘                                  │ JSON  /api/*
                                                          ▼
                                        ┌──────────────────────────────────────┐
                                        │       apps/console/frontend          │
                                        │  React → vite build → web-dist/      │
                                        │  screens/  components/  store.tsx    │
                                        └──────────────────────────────────────┘
```

## model-backend

`scripts/training.py` is the only entry point that writes a checkpoint. Everything downstream
loads a frozen one:

- `scripts/evaluation.py` — per-date predictions and forecast metrics.
- `scripts/run_backtest.py` — the trading matrix. Loads no model at all: it replays
  `output/evaluation/forecast_predictions.csv` through `utils/trade.py`.
- `thesis_pipeline/` — `contracts.py` pins each model's window, parameter count, checkpoint path
  and SHA-256; `inference.py` rebuilds the model from those specs, hashes the checkpoint bytes,
  and loads with `strict=True`; `evaluation.py` and `backtest.py` produce the aligned tables;
  `package.py` seals them into `evidence/`.
- `scripts/checkpoint_inference.py` — a stdin/stdout JSON worker wrapping `thesis_pipeline.inference`.
  This is what the console spawns.

Model classes are reached by string import: a training config names an architecture,
`configs/models/archs.yaml` maps it to a model YAML, and `utils/io_tools.py` imports the class in
that YAML's `target:` field. A static import graph will therefore show `pl_modules/cmamba_module.py`
and `cmamba_t_module.py` as unreferenced; they are not.

## console backend layers

| Layer | Responsibility | Modules |
|---|---|---|
| `routers/` | HTTP only: parse, call one service, translate domain errors. No pandas, no plotly. | `data`, `reproduce`, `predict`, `trading`, `architecture`, `_http` |
| `services/` | Business logic, one module per screen. Never imports FastAPI; raises `core.errors.ConsoleError`. | `data_service`, `reproduce_service`, `predict_service`, `trading_service`, `architecture_service` |
| `loaders/` | Data access: verify and read the evidence bundle, read frozen artifacts, drive the checkpoint worker. | `final_evidence`, `reproduction_evidence`, `forecast_robustness`, `checkpoint_inference` |
| `schemas/` | Pydantic request models. | `data`, `predict`, `trading` |
| `core/` | Path resolution, settings, domain error types. | `config`, `errors` |
| `vendor/cryptomamba_ui/` | Verbatim copy of the data/chart/trading/artifact modules. | 7 modules |

`routers/_http.py` maps `ConsoleError` subclasses to status codes — 400 invalid input, 413 upload
too large, 502 upstream failure, 503 evidence unavailable. Anything else propagates, so an
unexpected defect appears as a 500 instead of being reported as the caller's fault.

## Evidence integrity

`loaders/final_evidence.py` reads `evidence/SHA256SUMS` once, hashes every listed file, and
thereafter serves only manifest-covered paths. A missing manifest, a hash mismatch, an unlisted
extra file, or a symlink makes the dependent screen report `NOT_READY` with the reason. There is no
unverified fallback.

`model/checkpoint_provenance.json` inside that bundle pins both checkpoints by path, byte count and
SHA-256, so the artifacts and the weights that produced them are covered by the same manifest.

## Environment split

The console backend carries no PyTorch. Predict-screen inference runs as a bounded subprocess in
`model-backend/.venv`, with a timeout and capped stdout/stderr, and the console validates the
worker's JSON response before serving it — a malformed or incomplete response is an error, never a
partially rendered prediction.

CPU is sufficient for that worker: `models/cmamba.py` falls back to a pure-PyTorch
`selective_scan_ref` when the CUDA kernels are absent. Without the `model-backend` venv at all, the
Predict screen reports the worker's failure and the other four screens are unaffected.

## Provenance

Where each part came from before the monorepo:

| Path | Source |
|---|---|
| `apps/model-backend` | `Crypto-Mamba-BE` @ `thesis/pre-monorepo-snapshot` |
| `apps/console` | `Crypto-Mamba-Console` @ `thesis/final-console-snapshot` |
| `apps/console/backend/src/console_api/vendor/cryptomamba_ui` | `Crypto-Mamba-FE` — 7 modules, verbatim copy, only intra-package imports rewritten |
| `evidence/` | the `final/` bundle of `cryptomamba-thesis-evidence`, flattened in |
