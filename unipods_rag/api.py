from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from .config import Settings
from .schemas import (
    AskRequest,
    ChatMessage,
    DigestRequest,
    DigestResponse,
    Filters,
    IngestResult,
    Transcript,
    WhatsAppAskRequest,
    WhatsAppAskResponse,
    WhatsAppIngestMessage,
)
from .service import RagService


def _check_key(expected: str | None, provided: str | None) -> None:
    if not expected:
        return

    if not provided or not secrets.compare_digest(
        provided.encode(),
        expected.encode(),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


def create_router(
    service: RagService,
    settings: Settings | None = None,
) -> APIRouter:

    s = settings or service.s

    def query_auth(
        x_api_key: str | None = Header(default=None),
    ) -> None:
        _check_key(s.api_key, x_api_key)

    def ingest_auth(
        x_api_key: str | None = Header(default=None),
    ) -> None:
        _check_key(
            s.ingest_api_key or s.api_key,
            x_api_key,
        )

    router = APIRouter(
        prefix="/rag",
        tags=["rag"],
    )

    # ---------------------------------------------------------
    # WhatsApp message ingestion
    # ---------------------------------------------------------

    @router.post(
        "/ingest/messages",
        response_model=IngestResult,
        dependencies=[Depends(ingest_auth)],
    )
    def ingest_messages(
        payload: WhatsAppIngestMessage,
    ) -> IngestResult:

        message = ChatMessage(
            message_id=payload.messageId,
            channel=payload.groupId,
            author=payload.sender,
            text=payload.text,
            timestamp=payload.timestamp,
        )

        return service.ingest_messages([message])

    # ---------------------------------------------------------
    # Call transcript ingestion
    # ---------------------------------------------------------

    @router.post(
        "/ingest/transcript",
        response_model=IngestResult,
        dependencies=[Depends(ingest_auth)],
    )
    def ingest_transcript(
        payload: Transcript,
    ) -> IngestResult:

        return service.ingest_transcript(payload)

    # ---------------------------------------------------------
    # WhatsApp question
    # ---------------------------------------------------------

    @router.post(
        "/ask",
        response_model=WhatsAppAskResponse,
        dependencies=[Depends(query_auth)],
    )
    def ask(
        payload: WhatsAppAskRequest,
    ) -> WhatsAppAskResponse:

        internal_request = AskRequest(
            question=payload.question,
            filters=Filters(
                channel=payload.jid,
            ),
        )

        result = service.ask(internal_request)

        return WhatsAppAskResponse(
            jid=payload.jid,
            sender=payload.sender,
            answer=result.answer,
        )

    # ---------------------------------------------------------
    # Digest
    # ---------------------------------------------------------

    @router.post(
        "/digest",
        response_model=DigestResponse,
        dependencies=[Depends(query_auth)],
    )
    def digest(
        payload: DigestRequest,
    ) -> DigestResponse:

        return service.digest(payload)

    # ---------------------------------------------------------
    # Stats
    # ---------------------------------------------------------

    @router.get(
        "/stats",
        dependencies=[Depends(query_auth)],
    )
    def stats() -> dict:

        return service.stats()

    return router