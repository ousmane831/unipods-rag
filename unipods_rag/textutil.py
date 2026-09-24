"""Outils texte partagés (normalisation, mots-clés, langue, découpage)."""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+")
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

STOPWORDS = frozenset(
    """a an the and or of to in on at for is are was were be been it this that these those with as by from
    what who when where which how did do does has have had can could you your i we they he she our their
    any anyone someone about into than then so if not no yes there here us me my
    le la les un une des du de et ou en dans sur au aux pour est sont etait etaient etre ce cette ces avec
    par que qui quoi quand ou quel quelle quels quelles comment a ont avait avez avons nous vous ils elles
    je tu il elle on mon ma mes ton ta tes son sa ses notre votre leur leurs pas ne y se plus mais donc
    quelqu un quelquun""".split()
)

_FR_HINTS = frozenset(
    "le la les des une est sont quel quelle quels quelles comment pour dans avec qui que quand nous vous ont hier lien réunion appel".split()
)
_EN_HINTS = frozenset("the what who when where which how did does has have is are was for with about anyone yesterday link call meeting".split())


def fold(text: str) -> str:
    """Minuscules, sans accents, apostrophes normalisées."""
    text = unicodedata.normalize("NFKD", text.replace("\u2019", "'"))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(fold(text)) if len(w) > 1 and w not in STOPWORDS]


def stem(token: str) -> str:
    """Racine grossière : suffisant pour rapprocher submit / submission / submitted."""
    return token[:5] if len(token) >= 5 else token


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    fr = sum(w in _FR_HINTS for w in words)
    en = sum(w in _EN_HINTS for w in words)
    return "fr" if fr > en else "en"


def extract_links(text: str) -> list[str]:
    seen: list[str] = []
    for m in URL_RE.findall(text):
        url = m.rstrip(".,;:!?")
        if url not in seen:
            seen.append(url)
    return seen


def split_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """Découpe un long texte sur des frontières de phrases/mots, avec recouvrement."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "), window.rfind("\n"))
            if cut < max_chars * 0.5:
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]
