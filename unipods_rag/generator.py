
"""Génération de la réponse et du digest.

- AnthropicGenerator  : Claude via l'API Anthropic.
- LocalGenerator      : modèle local via Ollama (ex. qwen3:4b).
- ExtractiveGenerator : sans LLM. Renvoie les passages les plus pertinents
                        tels quels.
"""

from __future__ import annotations
import re
import logging
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from . import prompts
from .config import Settings
from .retriever import Hit, keyword_overlap
from .textutil import tokens

log = logging.getLogger(__name__)


class Generator(Protocol):
    name: str

    def answer(
        self,
        question: str,
        hits: list[Hit],
        relaxed: bool,
        tz: ZoneInfo,
        language: str,
    ) -> str: ...

    def digest(
        self,
        chunk_texts: list[str],
        period_label: str,
        language: str,
        max_chars: int,
    ) -> str: ...


def _batches(items: list[str], max_chars: int) -> list[list[str]]:
    batches: list[list[str]] = []
    cur: list[str] = []
    size = 0

    for it in items:
        if cur and size + len(it) > max_chars:
            batches.append(cur)
            cur, size = [], 0

        cur.append(it)
        size += len(it) + 2

    if cur:
        batches.append(cur)

    return batches


class AnthropicGenerator:
    """Générateur utilisant Claude via l'API Anthropic."""

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )
        self.model = settings.llm_model
        self.name = f"anthropic:{self.model}"

    def _call(
        self,
        system: str,
        user: str,
        max_tokens: int = 900,
    ) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": user,
                }
            ],
        )

        return "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", "") == "text"
        ).strip()

    def answer(
        self,
        question,
        hits,
        relaxed,
        tz,
        language,
    ) -> str:
        context = prompts.build_context(hits, tz)

        return self._call(
            prompts.SYSTEM_PROMPT,
            prompts.build_user_prompt(
                question,
                context,
                relaxed,
            ),
        )

    def digest(
        self,
        chunk_texts,
        period_label,
        language,
        max_chars,
    ) -> str:
        batches = _batches(chunk_texts, max_chars)

        if len(batches) == 1:
            user = prompts.build_digest_prompt(
                period_label,
                "\n\n".join(batches[0]),
                language,
            )

            return self._call(
                prompts.DIGEST_SYSTEM_PROMPT,
                user,
            )

        # Beaucoup de matière :
        # résumé par lots puis fusion (map-reduce).
        partials = [
            self._call(
                prompts.DIGEST_SYSTEM_PROMPT,
                prompts.build_digest_prompt(
                    f"{period_label} (part {i}/{len(batches)})",
                    "\n\n".join(batch),
                    language,
                ),
            )
            for i, batch in enumerate(batches, 1)
        ]

        return self._call(
            prompts.MERGE_SYSTEM_PROMPT,
            prompts.build_merge_prompt(
                period_label,
                partials,
                language,
            ),
        )


class LocalGenerator:
    """Générateur utilisant un modèle local via Ollama."""

    def __init__(self, settings: Settings) -> None:
        import ollama

        self._client = ollama.Client(
            host="http://localhost:11434"
        )

        self.model = settings.local_model
        self.name = f"ollama:{self.model}"

    def _call(
    self,
    system: str,
    user: str,
    max_tokens: int = 900,
    ) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            options={
                "num_predict": max_tokens,
                "temperature": 0.1,
            },
            think=False,
        )

        content = (response["message"]["content"] or "").strip()

        # Certaines versions/configurations de Qwen3 renvoient encore
        # le raisonnement dans content avant </think>.
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()

        return content
    def answer(
        self,
        question,
        hits,
        relaxed,
        tz,
        language,
    ) -> str:
        context = prompts.build_context(hits, tz)

        return self._call(
            prompts.SYSTEM_PROMPT,
            prompts.build_user_prompt(
                question,
                context,
                relaxed,
            ),
        )

    def digest(
        self,
        chunk_texts,
        period_label,
        language,
        max_chars,
    ) -> str:
        batches = _batches(chunk_texts, max_chars)

        if len(batches) == 1:
            user = prompts.build_digest_prompt(
                period_label,
                "\n\n".join(batches[0]),
                language,
            )

            return self._call(
                prompts.DIGEST_SYSTEM_PROMPT,
                user,
            )

        # Beaucoup de matière :
        # résumé par lots puis fusion (map-reduce).
        partials = [
            self._call(
                prompts.DIGEST_SYSTEM_PROMPT,
                prompts.build_digest_prompt(
                    f"{period_label} (part {i}/{len(batches)})",
                    "\n\n".join(batch),
                    language,
                ),
            )
            for i, batch in enumerate(batches, 1)
        ]

        return self._call(
            prompts.MERGE_SYSTEM_PROMPT,
            prompts.build_merge_prompt(
                period_label,
                partials,
                language,
            ),
        )


def best_lines(
    text: str,
    question: str,
    n: int = 2,
) -> list[str]:
    body = [
        line
        for line in text.splitlines()[1:]
        if line.strip()
    ]

    q = tokens(question)

    ranked = sorted(
        range(len(body)),
        key=lambda i: -keyword_overlap(q, body[i]),
    )[:n]

    return [
        body[i]
        for i in sorted(ranked)
    ]


class ExtractiveGenerator:
    """Générateur sans LLM."""

    name = "extractive"

    def answer(
        self,
        question,
        hits,
        relaxed,
        tz,
        language,
    ) -> str:
        head = {
            "en": (
                "No language model is configured, "
                "so here are the most relevant passages:"
            ),
            "fr": (
                "Aucun modèle de langage n'est configuré ; "
                "voici les passages les plus pertinents :"
            ),
        }[language]

        widened = {
            "en": (
                "Nothing matched the requested period, "
                "so these come from other dates."
            ),
            "fr": (
                "Rien ne correspondait à la période demandée ; "
                "ces passages viennent d'autres dates."
            ),
        }[language]

        lines = [head] + (
            [widened] if relaxed else []
        )

        for i, hit in enumerate(hits[:3], 1):
            when = datetime.fromtimestamp(
                hit.metadata.get("ts_start", 0),
                tz,
            ).strftime("%Y-%m-%d")

            snippet = " / ".join(
                best_lines(
                    hit.text,
                    question,
                )
            )

            lines.append(
                f"- [{i}] "
                f"{hit.metadata.get('title', '')} "
                f"({when}): {snippet}"
            )

        return "\n".join(lines)

    def digest(
        self,
        chunk_texts,
        period_label,
        language,
        max_chars,
    ) -> str:
        head = {
            "en": (
                "No language model is configured, "
                "so this is a raw outline of the period:"
            ),
            "fr": (
                "Aucun modèle de langage n'est configuré ; "
                "voici un aperçu brut de la période :"
            ),
        }[language]

        lines = [head]

        for text in chunk_texts[-8:]:
            parts = text.splitlines()
            first = (
                parts[1]
                if len(parts) > 1
                else ""
            )

            lines.append(
                f"- {parts[0]} {first[:160]}"
            )

        return "\n".join(lines)


def get_generator(settings: Settings) -> Generator:
    """Retourne le générateur selon LLM_BACKEND."""

    backend = settings.llm_backend

    # Mode sans LLM
    if backend == "extractive":
        return ExtractiveGenerator()

    # Mode LLM local via Ollama
    if backend == "local":
        return LocalGenerator(settings)

    # Mode Claude
    if backend == "anthropic" or (
        backend == "auto"
        and settings.anthropic_api_key
    ):
        if not settings.anthropic_api_key:
            raise ValueError(
                "LLM_BACKEND=anthropic demande "
                "ANTHROPIC_API_KEY."
            )

        return AnthropicGenerator(settings)

    # Mode automatique sans clé Anthropic
    if backend == "auto":
        log.warning(
            "Pas de ANTHROPIC_API_KEY : "
            "réponses en mode extractif "
            "(sans LLM)."
        )

        return ExtractiveGenerator()

    raise ValueError(
        f"LLM_BACKEND inconnu : {backend!r}"
    )
