"""Embeddings.

- HashEmbedder            : hors-ligne, sans téléchargement, déterministe. Sert au développement et aux
                            tests. Il capte le vocabulaire commun, PAS le sens : à ne pas utiliser en production.
- SentenceTransformerEmbedder : vrais embeddings sémantiques et multilingues (FR/EN) pour la production.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Protocol

from .config import Settings
from .textutil import tokens

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokens(text)
        feats: list[tuple[str, float]] = []
        for t in toks:
            feats.append((t, 1.0))
            padded = f"#{t}#"
            feats.extend((padded[i : i + 3], 0.25) for i in range(len(padded) - 2))
        feats.extend((f"{a}_{b}", 0.5) for a, b in zip(toks, toks[1:]))
        for feat, weight in feats:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += weight if (h >> 63) & 1 else -weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Le backend 'sentence-transformers' demande : pip install -r requirements-ml.txt"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self._e5 = "e5" in model_name.lower()  # les modèles E5 attendent des préfixes

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        if self._e5:
            text = f"query: {text}"
        return self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if settings.embedding_backend == "hash":
        log.warning(
            "Embedder 'hash' actif : mode développement. Pour la production, "
            "EMBEDDING_BACKEND=sentence-transformers."
        )
        return HashEmbedder()
    raise ValueError(f"EMBEDDING_BACKEND inconnu : {settings.embedding_backend!r}")
