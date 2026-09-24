"""Journal des sources brutes (SQLite).

Pourquoi ? Les webhooks livrent les messages un par un. Pour reconstruire une fenêtre de
conversation complète (et pouvoir ré-indexer si on change de modèle d'embedding), on garde
le texte d'origine ici. Le vecteur-store ne contient que la version indexée.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .schemas import ChatMessage, Transcript


class SourceLog:
    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS messages(
                    message_id TEXT PRIMARY KEY, channel TEXT NOT NULL, author TEXT NOT NULL,
                    text TEXT NOT NULL, ts INTEGER NOT NULL)"""
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_chan_ts ON messages(channel, ts)")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS transcripts(
                    call_id TEXT PRIMARY KEY, channel TEXT NOT NULL, title TEXT NOT NULL,
                    started_at INTEGER NOT NULL, text TEXT NOT NULL)"""
            )

    # ------------------------------------------------------------ messages
    def upsert_messages(self, msgs: list[ChatMessage]) -> None:
        rows = [(m.message_id, m.channel, m.author, m.text, int(m.timestamp.timestamp())) for m in msgs]
        with self._lock, self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?)", rows)

    def window(self, channel: str, start_ts: int, end_ts: int) -> list[ChatMessage]:
        """Messages d'un canal dans [start_ts, end_ts), triés par heure."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT message_id, channel, author, text, ts FROM messages "
                "WHERE channel=? AND ts>=? AND ts<? ORDER BY ts, message_id",
                (channel, start_ts, end_ts),
            ).fetchall()
        return [
            ChatMessage(message_id=r[0], channel=r[1], author=r[2], text=r[3],
                        timestamp=datetime.fromtimestamp(r[4], tz=timezone.utc))
            for r in rows
        ]

    def all_windows(self, window_seconds: int) -> list[tuple[str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT channel, ts - (ts % ?) FROM messages", (window_seconds,)
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ------------------------------------------------------------ transcripts
    def upsert_transcript(self, t: Transcript) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO transcripts VALUES (?,?,?,?,?)",
                (t.call_id, t.channel, t.title, int(t.started_at.timestamp()), t.text),
            )

    def all_transcripts(self) -> list[Transcript]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT call_id, channel, title, started_at, text FROM transcripts ORDER BY started_at"
            ).fetchall()
        return [
            Transcript(call_id=r[0], channel=r[1], title=r[2],
                       started_at=datetime.fromtimestamp(r[3], tz=timezone.utc), text=r[4])
            for r in rows
        ]

    # ------------------------------------------------------------ stats
    def count_messages(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    def count_transcripts(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
