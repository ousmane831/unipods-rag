"""Découpage en chunks.

Chat  : on regroupe les messages par fenêtre de temps fixe (1 h par défaut) puis par taille.
        Une fenêtre fixe rend les identifiants de chunks déterministes : ré-ingérer un message,
        ou en recevoir un nouveau dans la même heure, reconstruit proprement la même fenêtre.
Appel : on regroupe les répliques consécutives jusqu'à la taille max, avec un léger recouvrement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from .schemas import ChatMessage, Transcript
from .textutil import extract_links, split_text

_OFFSET_RE = re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def window_start(ts: int, minutes: int) -> int:
    size = minutes * 60
    return ts - ts % size


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------- chat
def chunk_chat_window(
    channel: str,
    msgs: list[ChatMessage],
    window_start_ts: int,
    tz: ZoneInfo,
    max_chars: int = 1200,
) -> list[Chunk]:
    lines: list[tuple[ChatMessage, str]] = []
    for m in msgs:
        text = _clean(m.text)
        if not text:
            continue
        local = m.timestamp.astimezone(tz)
        prefix = f"{m.author} ({local:%H:%M}): "
        # Un message très long est découpé, chaque morceau garde son auteur.
        for part in split_text(text, max_chars - len(prefix), overlap=0):
            lines.append((m, prefix + part))
    if not lines:
        return []

    groups: list[list[tuple[ChatMessage, str]]] = []
    cur: list[tuple[ChatMessage, str]] = []
    size = 0
    for item in lines:
        if cur and size + len(item[1]) + 1 > max_chars:
            groups.append(cur)
            cur, size = [], 0
        cur.append(item)
        size += len(item[1]) + 1
    if cur:
        groups.append(cur)

    window_id = f"chat:{channel}:{window_start_ts}"
    chunks: list[Chunk] = []
    for i, group in enumerate(groups):
        first, last = group[0][0].timestamp, group[-1][0].timestamp
        f_loc, l_loc = first.astimezone(tz), last.astimezone(tz)
        header = f"[Chat #{channel} | {f_loc:%Y-%m-%d} {f_loc:%H:%M}-{l_loc:%H:%M}]"
        body = "\n".join(line for _, line in group)
        authors = sorted({m.author for m, _ in group})
        text = f"{header}\n{body}"
        chunks.append(
            Chunk(
                id=f"{window_id}:{i}",
                text=text,
                metadata={
                    "source_type": "chat",
                    "channel": channel,
                    "title": f"#{channel}",
                    "ts_start": int(first.timestamp()),
                    "ts_end": int(last.timestamp()),
                    "date": f"{f_loc:%Y-%m-%d}",
                    "ref_id": window_id,
                    "window_id": window_id,
                    "has_link": bool(extract_links(body)),
                    "authors": ", ".join(authors),
                    "n_messages": len({m.message_id for m, _ in group}),
                },
            )
        )
    return chunks


# ---------------------------------------------------------------------------- appels
def chunk_transcript(
    t: Transcript,
    tz: ZoneInfo,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    turns: list[tuple[str | None, str]] = []  # (offset, réplique)
    for raw in t.text.splitlines():
        line = _clean(raw)
        if not line:
            continue
        m = _OFFSET_RE.match(line)
        offset = m.group(1) if m else None
        body = line[m.end():] if m else line
        for part in split_text(body, max_chars, overlap=0):
            turns.append((offset, part))
    if not turns:
        return []

    groups: list[list[tuple[str | None, str]]] = []
    cur: list[tuple[str | None, str]] = []
    size = 0
    for turn in turns:
        if cur and size + len(turn[1]) + 1 > max_chars:
            groups.append(cur)
            last = cur[-1]
            # recouvrement : on reporte la dernière réplique si elle est courte
            cur, size = ([last], len(last[1]) + 1) if len(last[1]) <= overlap_chars else ([], 0)
        cur.append(turn)
        size += len(turn[1]) + 1
    if cur:
        groups.append(cur)

    started = t.started_at.astimezone(tz)
    n = len(groups)
    chunks: list[Chunk] = []
    for i, group in enumerate(groups):
        offset = next((o for o, _ in group if o), None)
        part = f" | part {i + 1}/{n}" if n > 1 else ""
        at = f" | at {offset}" if offset else ""
        header = f"[Call: {t.title} | {started:%Y-%m-%d %H:%M}{part}{at}]"
        body = "\n".join(text for _, text in group)
        speakers = sorted({text.split(":", 1)[0] for _, text in group if ":" in text[:40]})
        chunks.append(
            Chunk(
                id=f"call:{t.call_id}:{i}",
                text=f"{header}\n{body}",
                metadata={
                    "source_type": "call",
                    "channel": t.channel,
                    "title": t.title,
                    "ts_start": int(t.started_at.timestamp()),
                    "ts_end": int(t.started_at.timestamp()),
                    "date": f"{started:%Y-%m-%d}",
                    "ref_id": t.call_id,
                    "call_id": t.call_id,
                    "has_link": bool(extract_links(body)),
                    "authors": ", ".join(speakers),
                    "n_messages": 0,
                    "offset": offset or "",
                },
            )
        )
    return chunks
