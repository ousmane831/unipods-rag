"""Vector store : interface unique, deux implémentations.

- ChromaStore : persistant (fichiers locaux), utilisé en dev et en démo.
- MemoryStore : en mémoire, pour les tests unitaires.
Passer à Pinecone plus tard = écrire une 3e classe qui respecte VectorStore, rien d'autre ne change.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .schemas import Filters


@dataclass
class StoredChunk:
    id: str
    text: str
    metadata: dict
    score: float | None = None  # similarité cosinus 0..1 (None pour get())


def _epoch(dt) -> int:
    return int(dt.timestamp())


def matches(meta: dict, f: Filters | None) -> bool:
    if f is None:
        return True
    if f.source_type and meta.get("source_type") != f.source_type.value:
        return False
    if f.channel and meta.get("channel") != f.channel:
        return False
    ts = meta.get("ts_start", 0)
    if f.since and ts < _epoch(f.since):
        return False
    if f.until and ts >= _epoch(f.until):
        return False
    return True


def chroma_where(f: Filters | None) -> dict | None:
    if f is None:
        return None
    conds: list[dict] = []
    if f.source_type:
        conds.append({"source_type": f.source_type.value})
    if f.channel:
        conds.append({"channel": f.channel})
    if f.since:
        conds.append({"ts_start": {"$gte": _epoch(f.since)}})
    if f.until:
        conds.append({"ts_start": {"$lt": _epoch(f.until)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None: ...

    @abstractmethod
    def query(self, embedding: list[float], k: int, filters: Filters | None = None) -> list[StoredChunk]: ...

    @abstractmethod
    def get(self, filters: Filters | None = None, limit: int = 1000) -> list[StoredChunk]: ...

    @abstractmethod
    def delete_where(self, key: str, value: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


class MemoryStore(VectorStore):
    def __init__(self) -> None:
        self._items: dict[str, tuple[str, list[float], dict]] = {}

    def upsert(self, ids, texts, embeddings, metadatas) -> None:
        for i, t, e, m in zip(ids, texts, embeddings, metadatas):
            self._items[i] = (t, e, m)

    def query(self, embedding, k, filters=None):
        scored = []
        for i, (t, e, m) in self._items.items():
            if matches(m, filters):
                sim = sum(a * b for a, b in zip(embedding, e))
                scored.append(StoredChunk(i, t, m, max(0.0, min(1.0, sim))))
        return sorted(scored, key=lambda c: c.score or 0, reverse=True)[:k]

    def get(self, filters=None, limit=1000):
        return [StoredChunk(i, t, m) for i, (t, _, m) in self._items.items() if matches(m, filters)][:limit]

    def delete_where(self, key, value) -> None:
        for i in [i for i, (_, _, m) in self._items.items() if m.get(key) == value]:
            del self._items[i]

    def count(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items.clear()


class ChromaStore(VectorStore):
    _BATCH = 500

    def __init__(self, path: str, collection: str, embedder_name: str = "default") -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))
        # Le nom du modèle d'embedding fait partie du nom de collection : changer de modèle ne
        # mélange jamais deux espaces vectoriels incompatibles (voir scripts/reindex.py).
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", embedder_name).strip("-")[:40]
        self._name = f"{collection}__{slug}"[:63]
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})

    def upsert(self, ids, texts, embeddings, metadatas) -> None:
        for s in range(0, len(ids), self._BATCH):
            e = s + self._BATCH
            self._col.upsert(ids=ids[s:e], documents=texts[s:e], embeddings=embeddings[s:e], metadatas=metadatas[s:e])

    def query(self, embedding, k, filters=None):
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(k, n),
            where=chroma_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(StoredChunk(i, doc, meta, max(0.0, min(1.0, 1.0 - dist))))
        return out

    def get(self, filters=None, limit=1000):
        res = self._col.get(where=chroma_where(filters), limit=limit, include=["documents", "metadatas"])
        return [StoredChunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def delete_where(self, key, value) -> None:
        self._col.delete(where={key: value})

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})
