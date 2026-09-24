"""Journal persistant des sources brutes dans PostgreSQL / Supabase.

Les webhooks livrent les messages un par un. On conserve les messages et
transcriptions d'origine afin de pouvoir reconstruire les fenêtres de
conversation et ré-indexer les données si le modèle d'embedding change.
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg

from .schemas import ChatMessage, Transcript


class SourceLog:
    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError(
                "DATABASE_URL est obligatoire pour utiliser SourceLog avec PostgreSQL."
            )

        self.database_url = database_url

    def _connect(self):
        """Ouvre une connexion PostgreSQL."""
        return psycopg.connect(self.database_url)

    # ------------------------------------------------------------ messages

    def upsert_messages(self, msgs: list[ChatMessage]) -> None:
        if not msgs:
            return

        rows = [
            (
                m.message_id,
                m.channel,
                m.author,
                m.text,
                int(m.timestamp.timestamp()),
            )
            for m in msgs
        ]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO public.rag_messages
                        (message_id, channel, author, text, ts)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (message_id)
                    DO UPDATE SET
                        channel = EXCLUDED.channel,
                        author = EXCLUDED.author,
                        text = EXCLUDED.text,
                        ts = EXCLUDED.ts
                    """,
                    rows,
                )

    def window(
        self,
        channel: str,
        start_ts: int,
        end_ts: int,
    ) -> list[ChatMessage]:
        """Messages d'un canal dans [start_ts, end_ts), triés par heure."""

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT message_id, channel, author, text, ts
                    FROM public.rag_messages
                    WHERE channel = %s
                      AND ts >= %s
                      AND ts < %s
                    ORDER BY ts, message_id
                    """,
                    (channel, start_ts, end_ts),
                )
                rows = cur.fetchall()

        return [
            ChatMessage(
                message_id=row[0],
                channel=row[1],
                author=row[2],
                text=row[3],
                timestamp=datetime.fromtimestamp(
                    row[4],
                    tz=timezone.utc,
                ),
            )
            for row in rows
        ]

    def all_windows(
        self,
        window_seconds: int,
    ) -> list[tuple[str, int]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT
                        channel,
                        ts - MOD(ts, %s)
                    FROM public.rag_messages
                    ORDER BY channel, ts - MOD(ts, %s)
                    """,
                    (window_seconds, window_seconds),
                )
                rows = cur.fetchall()

        return [(row[0], row[1]) for row in rows]

    # ------------------------------------------------------------ transcripts

    def upsert_transcript(self, t: Transcript) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.rag_transcripts
                        (call_id, channel, title, started_at, text)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (call_id)
                    DO UPDATE SET
                        channel = EXCLUDED.channel,
                        title = EXCLUDED.title,
                        started_at = EXCLUDED.started_at,
                        text = EXCLUDED.text
                    """,
                    (
                        t.call_id,
                        t.channel,
                        t.title,
                        int(t.started_at.timestamp()),
                        t.text,
                    ),
                )

    def all_transcripts(self) -> list[Transcript]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT call_id, channel, title, started_at, text
                    FROM public.rag_transcripts
                    ORDER BY started_at
                    """
                )
                rows = cur.fetchall()

        return [
            Transcript(
                call_id=row[0],
                channel=row[1],
                title=row[2],
                started_at=datetime.fromtimestamp(
                    row[3],
                    tz=timezone.utc,
                ),
                text=row[4],
            )
            for row in rows
        ]

    # ------------------------------------------------------------ stats

    def count_messages(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM public.rag_messages"
                )
                return cur.fetchone()[0]

    def count_transcripts(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM public.rag_transcripts"
                )
                return cur.fetchone()[0]