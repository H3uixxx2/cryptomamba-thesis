# console — CryptoMamba thesis demo

Five-screen demo of the thesis pipeline: **Data → Evaluation → Predict → Trading → Architecture**.
React frontend + thin FastAPI backend. The backend holds no model/data logic — it reads frozen
artifacts and the checksum-verified evidence bundle, and shells out to the `model-backend` venv
for real frozen-checkpoint inference. Every number on screen comes from an artifact or the API.

## Layout

```
console/
├── backend/
│   ├── pyproject.toml  requirements.txt
│   └── src/console_api/
│       ├── main.py              # FastAPI app factory + router mounts
│       ├── core/                # config.py (paths, env-overridable) + errors.py (domain errors)
│       ├── routers/             # HTTP only: parse -> call service -> map errors
│       ├── services/            # business logic, one module per screen (no FastAPI import)
│       ├── loaders/             # data access — evidence bundle, artifacts, checkpoint worker
│       │   ├── final_evidence.py         # verify-once SHA256SUMS reader for evidence/
│       │   ├── reproduction_evidence.py  # 350-day reproduction recompute
│       │   ├── forecast_robustness.py    # bootstrap interval reader
│       │   └── checkpoint_inference.py   # bounded subprocess adapter -> model-backend worker
│       ├── schemas/             # pydantic request models
│       ├── logic.py             # facade over the vendored cryptomamba_ui package
│       └── vendor/cryptomamba_ui/   # 7 modules vendored from the Streamlit repo (frozen copy)
├── frontend/                    # Vite + React 19 + TS + Tailwind + shadcn/ui + Plotly.js
│   └── src/{screens,components,lib,store}/
├── web-dist/                    # prebuilt frontend bundle, served at / by FastAPI
├── sample_data/                 # paper-split OHLCV fixture
└── scripts/{run_local.sh,build_ui.sh}
```

## External wiring (all env-overridable, defaults are monorepo-relative)

| Env var | Default | Used for |
|---|---|---|
| `CRYPTO_MAMBA_CORE_ROOT` | `../model-backend` | frozen `output/evaluation` + `output/reproduce_colab_train` artifacts |
| `CRYPTO_MAMBA_CORE_PYTHON` | `../model-backend/.venv/bin/python` | real checkpoint inference (Predict screen) |
| `CRYPTO_MAMBA_FINAL_EVIDENCE` | `../../evidence` | checksum-verified thesis-final bundle |
| `CRYPTO_MAMBA_API_URL` | — | optional live Colab/ngrok model API (pasted in the UI) |

## Run

```bash
# backend
python3 -m venv backend/.venv
./backend/.venv/bin/pip install -e backend
# frontend bundle is already in web-dist/; rebuild only if you change frontend/src:
#   ./scripts/build_ui.sh        (needs pnpm)
./scripts/run_local.sh           # http://127.0.0.1:8600
```

Frontend hot-reload dev loop (backend must run on 8600):

```bash
cd frontend && pnpm install && pnpm dev     # proxies /api -> :8600
```

## Test

```bash
cd backend && PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

81 contract tests: API response shapes, honest `NOT_READY` handling, evidence-integrity
fail-closed behavior, and static guards on thesis-critical frontend labels/diagrams.

## Notes

- `web-dist/` is committed so the demo runs without Node/pnpm.
- The `vendor/cryptomamba_ui/` copy is frozen; only its intra-package imports were rewritten.
  Upstream: <https://github.com/H3uixxx2/Crypto-Mamba-FE>.
- Predict-screen real inference needs the `model-backend` venv with PyTorch + Mamba, which
  build only on Linux + CUDA (Colab). Without it, that screen reports `NOT_READY` rather than
  faking a prediction.
