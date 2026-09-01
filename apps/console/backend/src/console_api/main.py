"""FastAPI application entry point.

Serves the static console under ``/`` and the JSON API under ``/api``. Holds no
business logic — every endpoint delegates to the reused ``cryptomamba_ui`` package.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core import config
from .routers import architecture as architecture_router
from .routers import data as data_router
from .routers import predict as predict_router
from .routers import reproduce as reproduce_router
from .routers import trading as trading_router

app = FastAPI(title="CryptoMamba Console", version="0.1.0")


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Serve the frontend with no-store so edits to JS/CSS take effect on reload
    (this is a single-user demo; correctness over micro-caching)."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/assets") or path.startswith("/js"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response

app.include_router(data_router.router)
app.include_router(reproduce_router.router)
app.include_router(predict_router.router)
app.include_router(trading_router.router)
app.include_router(architecture_router.router)


@app.get("/api/health")
def health() -> dict:
    return config.health()


# Static frontend. index.html is served explicitly at root; the rest is mounted.
@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.frontend_dir() / "index.html")


app.mount("/", StaticFiles(directory=config.frontend_dir()), name="web")
