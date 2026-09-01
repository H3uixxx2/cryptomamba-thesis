# Architecture

Two apps, one shared evidence bundle. Dependencies point one way: `console → model-backend`,
`console → evidence`. `model-backend` depends on nothing else in the repo.

```
                        ┌────────────────────────────┐
                        │  apps/model-backend        │
   train / evaluate ──▶ │  models/ pl_modules/       │
   (Colab, CUDA)        │  data_utils/ thesis_pipeline│
                        │                            │
                        │  writes:                   │
                        │   output/evaluation/*      │──────┐
                        │   output/reproduce_colab_* │      │
                        │   checkpoints/cmamba_v.ckpt │      │  read (files only)
                        └──────────┬─────────────────┘      │
                                   │ subprocess              ▼
                                   │ (real Predict     ┌───────────────────────────┐
                                   │  inference)       │  apps/console/backend      │
                                   └─────────────────▶ │  console_api/              │
                                                       │   routers → services      │
   ┌──────────────────┐   verify-once SHA256SUMS       │   loaders  → files/worker │
   │  evidence/           │◀────────────────────────────│   vendor/cryptomamba_ui   │
   │  SHA256SUMS + files │                             └──────────┬────────────────┘
   └──────────────────┘                                          │ JSON /api/*
                                                                 ▼
                                                       ┌───────────────────────────┐
                                                       │  apps/console/frontend     │
                                                       │  React → build → web-dist/ │
                                                       │  screens/ components/ lib/ │
                                                       └───────────────────────────┘
```

## console backend layers

| Layer | Responsibility | Modules |
|---|---|---|
| `routers/` | HTTP only: parse the request, call one service, translate domain errors. No pandas/plotly. | `data`, `reproduce`, `predict`, `trading`, `architecture`, `_http` |
| `loaders/` | data access — read frozen artifacts, verify the evidence bundle, drive the checkpoint worker | `final_evidence`, `reproduction_evidence`, `forecast_robustness`, `checkpoint_inference` |
| `vendor/cryptomamba_ui/` | frozen copy of the validated data/chart/trading/artifact logic reused from the Streamlit repo | 7 modules |
| `core/` | path resolution, settings, domain error types | `config`, `errors` |
| `services/` | business logic, one module per screen. Never imports FastAPI; raises `core.errors.ConsoleError`. | `data_service`, `reproduce_service`, `predict_service`, `trading_service`, `architecture_service` |
| `schemas/` | pydantic request models | `data`, `predict`, `trading` |

## Evidence integrity

`loaders/final_evidence.py` reads `evidence/SHA256SUMS` once, verifies every listed file's
hash, and thereafter serves only manifest-covered paths. A missing manifest, a hash mismatch, an
extra unlisted file, or a symlink makes the dependent screen report `NOT_READY` with an explicit
reason — it never falls back to unverified data.

## Environment split

The console backend intentionally carries **no** PyTorch/Mamba. Real frozen-checkpoint inference
for the Predict screen runs as a bounded subprocess in `model-backend/.venv` via
`scripts/checkpoint_inference.py` (which imports `thesis_pipeline.inference`). On a machine without
that environment the Predict screen degrades to `NOT_READY`; the other four screens are unaffected.
