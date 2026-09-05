# console — CryptoMamba demo

Five screens — **Data → Evaluation → Predict → Trading → Architecture** — served by a React
frontend over a thin FastAPI backend.

The backend holds no model or data logic. It reads frozen artifacts and the checksum-verified
evidence bundle, reuses the vendored `cryptomamba_ui` modules for transforms and charts, and
spawns the `model-backend` venv as a subprocess for real frozen-checkpoint inference. Every value
on screen traces to a file or to that worker; nothing is estimated or interpolated.

## Layout

```
console/
├── backend/
│   ├── pyproject.toml  requirements.txt
│   └── src/console_api/
│       ├── main.py              FastAPI app + router mounts + static frontend
│       ├── core/                config.py (paths, env-overridable), errors.py (domain errors)
│       ├── routers/             HTTP only: parse -> one service call -> map errors
│       ├── services/            business logic, one module per screen; no FastAPI import
│       ├── loaders/             data access
│       │   ├── final_evidence.py         verify-once SHA256SUMS reader
│       │   ├── reproduction_evidence.py  350-day metrics recomputed from pinned predictions
│       │   ├── forecast_robustness.py    bootstrap intervals
│       │   └── checkpoint_inference.py   bounded subprocess adapter -> model-backend worker
│       ├── schemas/             pydantic request models
│       ├── logic.py             single import point for the vendored package
│       └── vendor/cryptomamba_ui/   7 modules, verbatim copy
├── frontend/                    Vite + React 19 + TS + Tailwind + shadcn/ui + Plotly + Lightweight Charts
│   └── src/{screens,components,lib}/ + store.tsx
├── web-dist/                    prebuilt bundle, served at / by FastAPI
├── sample_data/                 paper-split OHLCV fixture
└── scripts/{run_local.sh,build_ui.sh}
```

Layer rule: `routers/` never touch pandas or plotly; `services/` never import FastAPI and raise
`core.errors.ConsoleError`, which `routers/_http.py` translates to a status code. Any other
exception propagates as a 500 rather than being mislabelled as a client error.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/data` · `POST /api/data/window` · `POST /api/data/upload` | split summary, causal window preview, CSV upload (25 MB cap) |
| `GET /api/reproduce` | 350-day reproduction + 304-date controlled comparison + paired tests |
| `POST /api/predict/checkpoint` | real inference in the `model-backend` venv |
| `POST /api/predict/live` | optional remote model API |
| `GET /api/predict/offline` | the frozen prediction, always labelled `offline` |
| `POST /api/trading/simulate` | one-day decision from one (current, predicted) pair |
| `GET /api/trading/backtest` · `GET /api/trading/replay` | multi-year chronological backtest, read from CSVs |
| `GET /api/architecture` | S5-Full model card + its metric rows |
| `GET /api/health` | liveness probe: resolved paths and whether each artifact root exists |

## Wiring

All env-overridable; defaults are monorepo-relative, so a plain clone works with no configuration.

| Env var | Default | Used for |
|---|---|---|
| `CRYPTO_MAMBA_CORE_ROOT` | `../model-backend` | frozen `output/` artifacts |
| `CRYPTO_MAMBA_CORE_PYTHON` | `../model-backend/.venv/bin/python` | the checkpoint-inference worker |
| `CRYPTO_MAMBA_FINAL_EVIDENCE` | `../../evidence` | checksum-verified bundle |
| `CRYPTO_MAMBA_API_URL` | — | optional live model API, normally pasted in the UI |

## Run

```bash
python3 -m venv backend/.venv
./backend/.venv/bin/python -m pip install --upgrade pip   # editable install needs pip >= 21.3
./backend/.venv/bin/pip install -e backend
./scripts/run_local.sh           # http://127.0.0.1:8600
```

`web-dist/` is committed, so no Node toolchain is needed to run the demo. Rebuild it only after
changing `frontend/src`:

```bash
./scripts/build_ui.sh            # needs pnpm
cd frontend && pnpm install && pnpm dev    # hot reload on :5273, proxies /api -> :8600
```

## Predict screen

Three paths, and `inference_type` is set by the path rather than by the caller:

- **Checkpoint** (default) — spawns `model-backend/.venv` to run `scripts/checkpoint_inference.py`
  against a frozen checkpoint. CPU is enough; no GPU and no network. If that venv is missing or
  broken the screen surfaces the worker's error, it does not fall back to another number.
- **Historical Replay** — reads the frozen per-date predictions.
- **Live HTTP** — optional; the only code path in the console that makes an outbound request.

Windows ending after the last trained date are flagged out-of-distribution.
