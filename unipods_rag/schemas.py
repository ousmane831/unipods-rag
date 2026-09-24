"""Contrat de données entre le backend (Gabriel), le moteur RAG (Ousmane) et l'interface.

C'est LE fichier à figer ensemble : tout changement ici doit être validé par les deux côtés.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    chat = "chat"
    call = "call"


def _aware(dt: datetime) -> datetime:
    """Un datetime sans fuseau est interprété comme UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- Ingestion (Gabriel -> Ousmane)
class ChatMessage(BaseModel):
    message_id: str = Field(min_length=1, description="Identifiant unique et stable du message")
    channel: str = "general"
    author: str = "unknown"
    text: str
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return _aware(v)


class MessagesPayload(BaseModel):
    messages: list[ChatMessage] = Field(max_length=2000)


class Transcript(BaseModel):
    call_id: str = Field(min_length=1, description="Identifiant unique de l'appel")
    title: str
    started_at: datetime
    text: str = Field(description='Une réplique par ligne : "Nom: texte" (préfixe [hh:mm:ss] optionnel)')
    channel: str = "calls"

    @field_validator("started_at")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return _aware(v)


class IngestResult(BaseModel):
    received: int
    chunks_indexed: int


# ---------------------------------------------------------------- Questions (Interface -> Ousmane)
class Filters(BaseModel):
    source_type: Optional[SourceType] = None
    channel: Optional[str] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None

    @field_validator("since", "until")
    @classmethod
    def _tz(cls, v: Optional[datetime]) -> Optional[datetime]:
        return _aware(v) if v else v


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=12)
    filters: Optional[Filters] = None
    language: Optional[str] = Field(default=None, pattern="^(fr|en)$")


class SourceRef(BaseModel):
    index: int
    source_type: SourceType
    channel: str
    title: str
    when: datetime
    excerpt: str
    score: float
    ref_id: str


class AskResponse(BaseModel):
    answer: str
    found: bool
    sources: list[SourceRef]
    filters_applied: list[str]
    relaxed: bool = False
    generator: str
    latency_ms: int


# ---------------------------------------------------------------- Digest
class DigestRequest(BaseModel):
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    channel: Optional[str] = None
    source_type: Optional[SourceType] = None
    language: str = Field(default="en", pattern="^(fr|en)$")

    @field_validator("since", "until")
    @classmethod
    def _tz(cls, v: Optional[datetime]) -> Optional[datetime]:
        return _aware(v) if v else v


class DigestStats(BaseModel):
    messages: int
    chunks: int
    authors: list[str]
    calls: list[str]


class DigestResponse(BaseModel):
    summary: str
    since: datetime
    until: datetime
    stats: DigestStats
    links: list[str]
    generator: str


class WhatsAppIngestMessage(BaseModel):
    messageId: str = Field(min_length=1)
    groupId: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    text: str = Field(min_length=1)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return _aware(v)


class WhatsAppAskRequest(BaseModel):
    jid: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    question: str = Field(min_length=2, max_length=1000)


class WhatsAppAskResponse(BaseModel):
    jid: str
    sender: str
    answer: str