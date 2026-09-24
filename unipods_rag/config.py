
"""Configuration centralisée.
Tout se règle par variables d'environnement (voir .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv est optionnel
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _opt(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------
    # Stockage
    # ------------------------------------------------------------------
    data_dir: str = "./data"
    collection: str = "unipods_chunks"

    # ------------------------------------------------------------------
    # Embeddings
    # "hash" (dev, hors-ligne)
    # ou "sentence-transformers" (production)
    # ------------------------------------------------------------------
    embedding_backend: str = "hash"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ------------------------------------------------------------------
    # Génération
    #
    # "auto"       : Claude si clé présente, sinon extractif
    # "anthropic"  : Claude via API Anthropic
    # "local"      : Ollama / modèle local
    # "extractive" : aucun LLM
    # ------------------------------------------------------------------
    llm_backend: str = "auto"

    # Modèle Anthropic
    llm_model: str = "claude-sonnet-5"

    # Modèle local Ollama
    local_model: str = "qwen3:4b"

    # Clé API Anthropic
    anthropic_api_key: str | None = None

    # ------------------------------------------------------------------
    # Temps
    # Sert à comprendre "hier", "cette semaine", etc.
    # ------------------------------------------------------------------
    timezone: str = "Africa/Dakar"

    # ------------------------------------------------------------------
    # Récupération
    # ------------------------------------------------------------------
    top_k: int = 6
    candidate_multiplier: int = 4
    min_score: float = 0.15
    confident_score: float = 0.30

    # ------------------------------------------------------------------
    # Découpage
    # ------------------------------------------------------------------
    chat_window_minutes: int = 60
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150

    # ------------------------------------------------------------------
    # Digest
    # ------------------------------------------------------------------
    digest_max_chars: int = 14000
    digest_max_chunks: int = 400

    # ------------------------------------------------------------------
    # Sécurité
    # ------------------------------------------------------------------
    api_key: str | None = None
    ingest_api_key: str | None = None
    cors_origins: tuple[str, ...] = ()
    
    # ------------------------------------------------------------------
    # Stockage
    # ------------------------------------------------------------------
    data_dir: str = "./data"
    collection: str = "unipods_chunks"

    # PostgreSQL / Supabase
    database_url: str | None = None
    cohere_api_key: str | None = None
    whatsapp_reply_url: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()

        return cls(
            # Stockage
            data_dir=os.getenv(
                "DATA_DIR",
                d.data_dir,
            ),
            collection=os.getenv(
                "COLLECTION",
                d.collection,
            ),
            database_url=_opt("DATABASE_URL"),
            cohere_api_key=_opt("COHERE_API_KEY"),
            whatsapp_reply_url=_opt("WHATSAPP_REPLY_URL"),
            # Embeddings
            embedding_backend=os.getenv(
                "EMBEDDING_BACKEND",
                d.embedding_backend,
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                d.embedding_model,
            ),

            # Génération
            llm_backend=os.getenv(
                "LLM_BACKEND",
                d.llm_backend,
            ),
            llm_model=os.getenv(
                "LLM_MODEL",
                d.llm_model,
            ),
            local_model=os.getenv(
                "LOCAL_MODEL",
                d.local_model,
            ),
            anthropic_api_key=_opt(
                "ANTHROPIC_API_KEY"
            ),

            # Temps
            timezone=os.getenv(
                "TIMEZONE",
                d.timezone,
            ),

            # Récupération
            top_k=int(
                os.getenv(
                    "TOP_K",
                    d.top_k,
                )
            ),
            candidate_multiplier=int(
                os.getenv(
                    "CANDIDATE_MULTIPLIER",
                    d.candidate_multiplier,
                )
            ),
            min_score=float(
                os.getenv(
                    "MIN_SCORE",
                    d.min_score,
                )
            ),
            confident_score=float(
                os.getenv(
                    "CONFIDENT_SCORE",
                    d.confident_score,
                )
            ),

            # Découpage
            chat_window_minutes=int(
                os.getenv(
                    "CHAT_WINDOW_MINUTES",
                    d.chat_window_minutes,
                )
            ),
            chunk_max_chars=int(
                os.getenv(
                    "CHUNK_MAX_CHARS",
                    d.chunk_max_chars,
                )
            ),
            chunk_overlap_chars=int(
                os.getenv(
                    "CHUNK_OVERLAP_CHARS",
                    d.chunk_overlap_chars,
                )
            ),

            # Digest
            digest_max_chars=int(
                os.getenv(
                    "DIGEST_MAX_CHARS",
                    d.digest_max_chars,
                )
            ),
            digest_max_chunks=int(
                os.getenv(
                    "DIGEST_MAX_CHUNKS",
                    d.digest_max_chunks,
                )
            ),

            # Sécurité
            api_key=_opt("API_KEY"),
            ingest_api_key=_opt("INGEST_API_KEY"),

            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "CORS_ORIGINS",
                    "",
                ).split(",")
                if origin.strip()
            ),
        )

    @property
    def chroma_path(self) -> str:
        return str(
            Path(self.data_dir) / "chroma"
        )

    @property
    def source_log_path(self) -> str:
        return str(
            Path(self.data_dir) / "sources.db"
        )