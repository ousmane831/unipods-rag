"""RagService : le point d'entrée unique du moteur.

    service = RagService()
    service.ingest_messages([...])      # appelé par le backend de Gabriel (webhook)
    service.ingest_transcript(t)        # idem, pour un appel
    service.ask(AskRequest(...))        # appelé par l'interface
    service.digest(DigestRequest(...))  # idem
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Callable
from zoneinfo import ZoneInfo

from . import prompts
from .chunking import Chunk, chunk_chat_window, chunk_transcript, window_start
from .config import Settings
from .embeddings import Embedder, get_embedder
from .generator import ExtractiveGenerator, Generator, get_generator
from .query_understanding import understand
from .retriever import Retriever
from .schemas import (
    AskRequest, AskResponse, ChatMessage, DigestRequest, DigestResponse, DigestStats, Filters,
    IngestResult, SourceRef, SourceType, Transcript,
)
from .source_log import SourceLog
from .textutil import detect_language, extract_links
from .vector_store import ChromaStore, VectorStore

log = logging.getLogger(__name__)


def _excerpt(text: str, limit: int = 600) -> str:
    """Coupe sur une limite de ligne ou de mot (jamais au milieu d'un mot) et signale la coupure."""
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    if cut < limit * 0.5:
        cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip() + "\u2026"


class RagService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        source_log: SourceLog | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.s = settings or Settings.from_env()
        self.tz = ZoneInfo(self.s.timezone)
        self.embedder = embedder or get_embedder(self.s)
        self.store = store or ChromaStore(self.s.chroma_path, self.s.collection, self.embedder.name)
        self.log = source_log or SourceLog(self.s.source_log_path)
        self.generator = generator or get_generator(self.s)
        self._fallback = ExtractiveGenerator()
        self.retriever = Retriever(self.store, self.embedder, self.s)
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------ ingestion
    def _add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.upsert([c.id for c in chunks], [c.text for c in chunks], vectors, [c.metadata for c in chunks])

    def _index_window(self, channel: str, ws: int) -> int:
        size = self.s.chat_window_minutes * 60
        msgs = self.log.window(channel, ws, ws + size)
        chunks = chunk_chat_window(channel, msgs, ws, self.tz, self.s.chunk_max_chars)
        self.store.delete_where("window_id", f"chat:{channel}:{ws}")  # la fenêtre est reconstruite en entier
        self._add(chunks)
        return len(chunks)

    def _index_transcript(self, t: Transcript) -> int:
        chunks = chunk_transcript(t, self.tz, self.s.chunk_max_chars, self.s.chunk_overlap_chars)
        self.store.delete_where("call_id", t.call_id)
        self._add(chunks)
        return len(chunks)

    def ingest_messages(self, messages: list[ChatMessage]) -> IngestResult:
        """Idempotent : renvoyer le même message (retry de webhook) ne crée aucun doublon."""
        msgs = [m for m in messages if m.text.strip()]
        if not msgs:
            return IngestResult(received=len(messages), chunks_indexed=0)
        self.log.upsert_messages(msgs)
        windows = {
            (m.channel, window_start(int(m.timestamp.timestamp()), self.s.chat_window_minutes)) for m in msgs
        }
        n = sum(self._index_window(ch, ws) for ch, ws in windows)
        return IngestResult(received=len(messages), chunks_indexed=n)

    def ingest_transcript(self, t: Transcript) -> IngestResult:
        self.log.upsert_transcript(t)
        return IngestResult(received=1, chunks_indexed=self._index_transcript(t))

    def reindex(self) -> IngestResult:
        """Reconstruit tout l'index depuis le journal des sources (après changement de modèle d'embedding)."""
        self.store.reset()
        n = 0
        for ch, ws in self.log.all_windows(self.s.chat_window_minutes * 60):
            n += self._index_window(ch, ws)
        transcripts = self.log.all_transcripts()
        for t in transcripts:
            n += self._index_transcript(t)
        return IngestResult(received=self.log.count_messages() + len(transcripts), chunks_indexed=n)

    # ------------------------------------------------------------------ questions
    def ask(self, req: AskRequest) -> AskResponse:
        t0 = perf_counter()
        now = self.now_fn()
        lang = req.language or detect_language(req.question)
        parsed = understand(req.question, now, self.tz)
        explicit = req.filters or Filters()
        k = req.top_k or self.s.top_k

        # Les filtres explicites de l'interface l'emportent sur ceux déduits de la phrase.
        filters = Filters(
            source_type=explicit.source_type or parsed.filters.source_type,
            channel=explicit.channel,
            since=explicit.since or parsed.filters.since,
            until=explicit.until or parsed.filters.until,
        )
        applied = list(parsed.notes)
        if explicit.source_type:
            applied.append(f"{explicit.source_type.value} only (selected)")
        if explicit.channel:
            applied.append(f"channel #{explicit.channel}")

        def run(f: Filters):
            return self.retriever.retrieve(
                req.question, f, k, wants_link=parsed.wants_link, prefers_recent=parsed.prefers_recent, now=now
            )

        hits = run(filters)
        best = hits[0].relevance if hits else 0.0
        relaxed = False
        inferred = bool(parsed.notes) and not (explicit.since or explicit.until)
        if inferred and best < self.s.confident_score:
            # Peu (ou rien) pour "hier" / "les appels" : on regarde sans ces filtres déduits.
            # On élargit si ça trouve enfin quelque chose de pertinent, ou nettement mieux (2x) ailleurs.
            wide = run(Filters(source_type=explicit.source_type, channel=explicit.channel,
                               since=explicit.since, until=explicit.until))
            top = wide[0].relevance if wide else 0.0
            if top >= self.s.min_score and (best < self.s.min_score or top >= 2 * best):
                hits, relaxed = wide, True
        hits = [h for h in hits if h.relevance >= self.s.min_score]

        if not hits:
            return AskResponse(
                answer=prompts.NOT_FOUND[lang], found=False, sources=[], filters_applied=applied,
                relaxed=False, generator=self.generator.name, latency_ms=int((perf_counter() - t0) * 1000),
            )

        generator = self.generator
        try:
            answer = generator.answer(req.question, hits, relaxed, self.tz, lang)
        except Exception:  # panne API, quota, réseau : on dégrade, on ne plante pas
            log.exception("Échec du générateur %s, repli extractif", generator.name)
            generator = self._fallback
            answer = generator.answer(req.question, hits, relaxed, self.tz, lang)

        sources = []
        for i, h in enumerate(hits, 1):
            body = "\n".join(h.text.splitlines()[1:]) or h.text
            sources.append(
                SourceRef(
                    index=i,
                    source_type=SourceType(h.metadata["source_type"]),
                    channel=h.metadata.get("channel", ""),
                    title=h.metadata.get("title", ""),
                    when=datetime.fromtimestamp(h.metadata["ts_start"], timezone.utc),
                    excerpt=_excerpt(body),
                    score=round(h.score, 3),
                    ref_id=h.metadata.get("ref_id", h.id),
                )
            )
        return AskResponse(
            answer=answer, found=True, sources=sources, filters_applied=applied, relaxed=relaxed,
            generator=generator.name, latency_ms=int((perf_counter() - t0) * 1000),
        )

    # ------------------------------------------------------------------ digest
    def digest(self, req: DigestRequest) -> DigestResponse:
        now = self.now_fn()
        until = req.until or now
        since = req.since or (until - timedelta(hours=24))
        lang = req.language
        f = Filters(source_type=req.source_type, channel=req.channel, since=since, until=until)
        chunks = sorted(self.store.get(f, limit=2000), key=lambda c: c.metadata.get("ts_start", 0))
        chunks = chunks[-self.s.digest_max_chunks:]

        stats = DigestStats(
            messages=sum(int(c.metadata.get("n_messages", 0)) for c in chunks),
            chunks=len(chunks),
            authors=sorted({a for c in chunks if c.metadata["source_type"] == "chat"
                            for a in c.metadata.get("authors", "").split(", ") if a}),
            calls=sorted({f"{c.metadata['title']} ({c.metadata['date']})" for c in chunks
                          if c.metadata["source_type"] == "call"}),
        )
        links: list[str] = []
        for c in chunks:
            for url in extract_links(c.text):
                if url not in links:
                    links.append(url)

        if not chunks:
            summary, gen_name = prompts.NO_ACTIVITY[lang], self.generator.name
        else:
            label = f"{since.astimezone(self.tz):%Y-%m-%d %H:%M} to {until.astimezone(self.tz):%Y-%m-%d %H:%M}"
            texts = [c.text for c in chunks]
            generator = self.generator
            try:
                summary = generator.digest(texts, label, lang, self.s.digest_max_chars)
            except Exception:
                log.exception("Échec du digest avec %s, repli extractif", generator.name)
                generator = self._fallback
                summary = generator.digest(texts, label, lang, self.s.digest_max_chars)
            gen_name = generator.name

        return DigestResponse(
            summary=summary, since=since, until=until, stats=stats, links=links[:20], generator=gen_name
        )

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        return {
            "messages": self.log.count_messages(),
            "calls": self.log.count_transcripts(),
            "chunks": self.store.count(),
            "embedder": self.embedder.name,
            "generator": self.generator.name,
            "timezone": self.s.timezone,
        }

   