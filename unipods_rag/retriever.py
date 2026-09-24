"""Récupération hybride.

Étape 1 : la recherche vectorielle ramène k x 4 candidats.
Étape 2 : on re-classe avec un score composite :
    pertinence = sémantique (sens) + mots-clés (noms propres, sigles, mots exacts),
    puis + récence + bonus "lien" pour départager.
Les chats parlent de noms, de dates et de liens : le sens seul rate souvent ces cas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from .config import Settings
from .embeddings import Embedder
from .schemas import Filters
from .textutil import stem, tokens
from .vector_store import VectorStore


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float  # classement final : pertinence + récence + bonus lien
    semantic: float
    keyword: float
    relevance: float = 0.0  # contenu seul (sens + mots-clés) : c'est elle qui sert de seuil "trouvé / pas trouvé"


def keyword_overlap(question_tokens: list[str], text: str) -> float:
    """Part des mots-clés de la question retrouvés dans le chunk.

    Mot exact = 1 point ; simple mot de même racine (share ~ shared) = 0,5 point.
    Sans cette nuance, "share" dans un message hors-sujet valait autant que "submission" dans le bon.
    """
    unique = set(question_tokens)
    if not unique:
        return 0.0
    chunk_tokens = set(tokens(text))
    chunk_stems = {stem(t) for t in chunk_tokens}
    score = 0.0
    for t in unique:
        if t in chunk_tokens:
            score += 1.0
        elif stem(t) in chunk_stems:
            score += 0.5
    return score / len(unique)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings) -> None:
        self.store, self.embedder, self.s = store, embedder, settings

    def retrieve(
        self,
        question: str,
        filters: Filters | None,
        top_k: int,
        *,
        wants_link: bool = False,
        prefers_recent: bool = False,
        now: datetime | None = None,
    ) -> list[Hit]:
        candidates = self.store.query(
            self.embedder.embed_query(question), top_k * self.s.candidate_multiplier, filters
        )
        q_tokens = tokens(question)
        now_ts = (now or datetime.now().astimezone()).timestamp()
        recency_weight = 0.15 if prefers_recent else 0.05

        hits: list[Hit] = []
        for c in candidates:
            sem = c.score or 0.0
            kw = keyword_overlap(q_tokens, c.text)
            age_days = max(0.0, (now_ts - c.metadata.get("ts_start", now_ts)) / 86400)
            recency = math.exp(-age_days / (14 if prefers_recent else 60))
            bonus = 0.12 if wants_link and c.metadata.get("has_link") else 0.0
            relevance = 0.6 * sem + 0.3 * kw
            # Récence et bonus lien ne servent qu'à départager : ils ne doivent jamais faire passer
            # un passage hors-sujet au-dessus du seuil.
            score = relevance + recency_weight * recency + bonus
            hits.append(Hit(c.id, c.text, c.metadata, score, sem, kw, relevance))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
