"""Mesure la qualité de la RÉCUPÉRATION (avant même de parler du LLM).

    python -m scripts.evaluate eval/questions.json

Format du fichier : une liste de
    {"question": "...", "expect_contains": ["mot ou lien attendu dans la bonne source"], "language": "en"}
Le test est réussi pour un rang k si l'un des k premiers passages contient TOUS les textes attendus.

Sorties : hit@1, hit@3, hit@k, MRR, et la liste des questions ratées (à examiner une par une).
C'est l'outil d'Adeyinka pour fixer un "seuil de qualité" et l'outil d'Ousmane pour régler
TOP_K, MIN_SCORE, CONFIDENT_SCORE et choisir le modèle d'embedding.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from unipods_rag.schemas import AskRequest
from unipods_rag.service import RagService
from unipods_rag.textutil import fold


def rank_of_match(sources, expected: list[str]) -> int | None:
    for s in sources:
        text = fold(s.excerpt)
        if all(fold(e) in text for e in expected):
            return s.index
    return None


def main(path: str) -> int:
    cases = json.load(open(path, encoding="utf-8"))
    service = RagService()
    service.now_fn = lambda: datetime.now(timezone.utc)
    ranks: list[int | None] = []
    misses = []
    for c in cases:
        r = service.ask(AskRequest(question=c["question"], language=c.get("language"), top_k=6))
        rank = rank_of_match(r.sources, c["expect_contains"])
        ranks.append(rank)
        if rank is None:
            misses.append((c["question"], c["expect_contains"], r.found))

    n = len(cases)
    hit = lambda k: sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"Questions : {n}")
    print(f"hit@1 = {hit(1):.0%}   hit@3 = {hit(3):.0%}   hit@6 = {hit(6):.0%}   MRR = {mrr:.2f}")
    for q, exp, found in misses:
        print(f"  RATÉE : {q!r}  attendu {exp}  ({'sources trouvées mais mauvaises' if found else 'aucune source'})")
    return 0 if not misses else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
