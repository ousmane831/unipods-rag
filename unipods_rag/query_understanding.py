"""Compréhension légère des questions, FR + EN, sans appel LLM.

Transforme "What were the action items from yesterday's call?" en :
  filtre source = appels, filtre date = hier (dans le fuseau du groupe).
Règles volontairement simples et prévisibles : facile à lire, à tester, à corriger.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .schemas import Filters, SourceType
from .textutil import fold

_YESTERDAY = re.compile(r"\b(yesterday|hier)\b")
_TODAY = re.compile(r"\b(today|aujourd'?\s?hui|this morning|ce matin|tonight|ce soir)\b")
_LAST_WEEK = re.compile(r"\b(last week|la semaine derniere|semaine passee|semaine derniere)\b")
_THIS_WEEK = re.compile(r"\b(this week|cette semaine)\b")
_LAST_N_DAYS = re.compile(r"\b(?:last|past)\s+(\d{1,3})\s+days\b|\b(\d{1,3})\s+derniers\s+jours\b")
_CALL = re.compile(
    r"\b(calls?|meetings?|reunions?|appels?|zoom|transcripts?|transcriptions?|recordings?|enregistrements?)\b"
)
_CHAT = re.compile(r"\b(in the chat|in the group|dans le chat|dans le groupe|whatsapp)\b")
_LINK = re.compile(r"\b(links?|liens?|urls?|forms?|formulaires?)\b")
_RECENT = re.compile(r"\b(latest|most recent|newest|recent|recente?s?|derniere?s?)\b|\blast (?!week|month|\d)")


@dataclass
class ParsedQuery:
    filters: Filters = field(default_factory=Filters)
    wants_link: bool = False
    prefers_recent: bool = False
    notes: list[str] = field(default_factory=list)  # description lisible des filtres déduits


def _midnight(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def understand(question: str, now: datetime, tz: ZoneInfo) -> ParsedQuery:
    q = fold(question)
    local_now = now.astimezone(tz)
    today = _midnight(local_now)
    out = ParsedQuery()

    since = until = None
    if _YESTERDAY.search(q):
        since, until = today - timedelta(days=1), today
        out.notes.append(f"yesterday ({since:%Y-%m-%d})")
    elif _LAST_WEEK.search(q):
        monday = today - timedelta(days=today.weekday())
        since, until = monday - timedelta(days=7), monday
        out.notes.append(f"last week ({since:%Y-%m-%d} to {(until - timedelta(days=1)):%Y-%m-%d})")
    elif _THIS_WEEK.search(q):
        since, until = today - timedelta(days=today.weekday()), local_now
        out.notes.append(f"this week (since {since:%Y-%m-%d})")
    elif (m := _LAST_N_DAYS.search(q)):
        n = int(m.group(1) or m.group(2))
        since, until = today - timedelta(days=n - 1), local_now
        out.notes.append(f"last {n} days")
    elif _TODAY.search(q):
        since, until = today, local_now + timedelta(minutes=1)
        out.notes.append(f"today ({today:%Y-%m-%d})")
    out.filters.since, out.filters.until = since, until

    if _CALL.search(q):
        out.filters.source_type = SourceType.call
        out.notes.append("calls only")
    elif _CHAT.search(q):
        out.filters.source_type = SourceType.chat
        out.notes.append("chat only")

    out.wants_link = bool(_LINK.search(q))
    out.prefers_recent = bool(_RECENT.search(q)) and since is None
    return out
