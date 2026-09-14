"""FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.exceptions import MolGNNError
from app.schemas.models import ErrorResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Convert chemical input (SMILES, drug name, or formula) into an "
        "interactive molecular graph and GNN-derived latent embedding."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(MolGNNError)
async def domain_error_handler(request: Request, exc: MolGNNError) -> JSONResponse:
    """Catch-all safety net for any domain exception that a route forgot to
    map explicitly — ensures the client never sees a raw 500 traceback."""
    logger.warning("Unhandled domain error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error_type=type(exc).__name__, message=str(exc)).model_dump(),
    )


app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/")
def root() -> dict:
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "api_prefix": settings.API_PREFIX,
    }
