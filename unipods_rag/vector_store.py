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

import psycopg
from psycopg.types.json import Jsonb




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


class PgVectorStore(VectorStore):
    """Vector store PostgreSQL + pgvector utilisé avec Supabase."""

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError(
                "DATABASE_URL est obligatoire pour utiliser PgVectorStore."
            )

        self.database_url = database_url

    def _connect(self):
        return psycopg.connect(self.database_url)

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        """Convertit un vecteur Python au format accepté par pgvector."""
        return "[" + ",".join(str(float(v)) for v in vector) + "]"

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return

        if not (
            len(ids)
            == len(texts)
            == len(embeddings)
            == len(metadatas)
        ):
            raise ValueError(
                "ids, texts, embeddings et metadatas "
                "doivent avoir la même longueur."
            )

        rows = []

        for chunk_id, text, embedding, metadata in zip(
            ids,
            texts,
            embeddings,
            metadatas,
        ):
            rows.append(
                (
                    chunk_id,
                    text,
                    metadata.get("source_type"),
                    metadata.get("channel"),
                    metadata.get("ts_start"),
                    metadata.get("ts_end"),
                    Jsonb(metadata),
                    self._vector_literal(embedding),
                )
            )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO public.rag_chunks (
                        id,
                        text,
                        source_type,
                        channel,
                        ts_start,
                        ts_end,
                        metadata,
                        embedding
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::extensions.vector
                    )
                    ON CONFLICT (id)
                    DO UPDATE SET
                        text = EXCLUDED.text,
                        source_type = EXCLUDED.source_type,
                        channel = EXCLUDED.channel,
                        ts_start = EXCLUDED.ts_start,
                        ts_end = EXCLUDED.ts_end,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    rows,
                )

    def query(
        self,
        embedding: list[float],
        k: int,
        filters: Filters | None = None,
    ) -> list[StoredChunk]:
        conditions: list[str] = []
        params: list = []

        if filters:
            if filters.source_type:
                conditions.append("source_type = %s")
                params.append(filters.source_type.value)

            if filters.channel:
                conditions.append("channel = %s")
                params.append(filters.channel)

            if filters.since:
                conditions.append("ts_start >= %s")
                params.append(_epoch(filters.since))

            if filters.until:
                conditions.append("ts_start < %s")
                params.append(_epoch(filters.until))

        where = ""

        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        vector = self._vector_literal(embedding)

        # Le même vecteur sert au calcul du score et au tri.
        query_params = [vector] + params + [vector, k]

        sql = f"""
            SELECT
                id,
                text,
                metadata,
                GREATEST(
                    0.0,
                    LEAST(
                        1.0,
                        1.0 - (
                            embedding
                            <=> %s::extensions.vector
                        )
                    )
                ) AS score
            FROM public.rag_chunks
            {where}
            ORDER BY
                embedding <=> %s::extensions.vector
            LIMIT %s
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, query_params)
                rows = cur.fetchall()

        return [
            StoredChunk(
                id=row[0],
                text=row[1],
                metadata=row[2],
                score=float(row[3]),
            )
            for row in rows
        ]

    def get(
        self,
        filters: Filters | None = None,
        limit: int = 1000,
    ) -> list[StoredChunk]:
        conditions: list[str] = []
        params: list = []

        if filters:
            if filters.source_type:
                conditions.append("source_type = %s")
                params.append(filters.source_type.value)

            if filters.channel:
                conditions.append("channel = %s")
                params.append(filters.channel)

            if filters.since:
                conditions.append("ts_start >= %s")
                params.append(_epoch(filters.since))

            if filters.until:
                conditions.append("ts_start < %s")
                params.append(_epoch(filters.until))

        where = ""

        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        params.append(limit)

        sql = f"""
            SELECT id, text, metadata
            FROM public.rag_chunks
            {where}
            ORDER BY ts_start
            LIMIT %s
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            StoredChunk(
                id=row[0],
                text=row[1],
                metadata=row[2],
            )
            for row in rows
        ]

    def delete_where(self, key: str, value: str) -> None:
        # Ces champs ont leur propre colonne.
        columns = {
            "channel",
            "source_type",
        }

        with self._connect() as conn:
            with conn.cursor() as cur:
                if key in columns:
                    cur.execute(
                        f"""
                        DELETE FROM public.rag_chunks
                        WHERE {key} = %s
                        """,
                        (value,),
                    )
                else:
                    # window_id, call_id, ref_id, etc.
                    # restent dans metadata JSONB.
                    cur.execute(
                        """
                        DELETE FROM public.rag_chunks
                        WHERE metadata ->> %s = %s
                        """,
                        (key, value),
                    )

    def count(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM public.rag_chunks"
                )
                return cur.fetchone()[0]

    def reset(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "TRUNCATE TABLE public.rag_chunks"
                )