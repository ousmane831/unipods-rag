"""Application autonome : API + interface web. Pour le développement et la démo.

    uvicorn unipods_rag.main:create_app --factory --reload

En production, c'est le backend de Gabriel qui porte l'API ; ce fichier n'est alors qu'un exemple d'assemblage.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import create_router
from .service import RagService

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app(service: RagService | None = None) -> FastAPI:
    service = service or RagService()
    app = FastAPI(title="UniPods Group Intelligence : moteur RAG", version="0.1.0")

    if service.s.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(service.s.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok", **{k: v for k, v in service.stats().items() if k in ("embedder", "generator")}}

    app.include_router(create_router(service))

    if FRONTEND_DIR.exists():  # monté en dernier : ne masque aucune route d'API
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
    return app
