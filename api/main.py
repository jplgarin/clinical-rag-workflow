"""FastAPI application entrypoint.

Run locally with::

    uvicorn api.main:app --reload

The web UI is served at ``/app`` and the JSON API lives under ``/api``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from core import __version__
from api.routes import registry, router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("clinical_rag")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _load_adapters() -> None:
    """Discover and register adapters at startup.

    The bundled Neuraxis example is registered when present. Real deployments
    typically replace this with their own registration, or drive it from
    configuration. Failures are logged but do not abort startup, so the
    service still answers /health while one domain is misconfigured.
    """
    try:
        from examples.adhd_neuraxis.adapter import NeuraxisAdapter

        registry.register(NeuraxisAdapter())
    except Exception:  # pragma: no cover - best-effort discovery
        logger.exception("could not register the Neuraxis example adapter")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting clinical-rag-workflow v%s", __version__)
    _load_adapters()
    logger.info("ready | domains=%s", registry.domains())
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Clinical RAG Workflow",
    description="Domain-agnostic clinical report generation with RAG.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Plain liveness check, mirrors ``/api/health`` status field."""
    return {"status": "ok", "version": __version__}


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/app/")


if WEB_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
