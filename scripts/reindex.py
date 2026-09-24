"""Reconstruit tout l'index depuis le journal des sources (sources.db).

À lancer après un changement de EMBEDDING_BACKEND / EMBEDDING_MODEL :

    python -m scripts.reindex
"""
from unipods_rag.service import RagService

if __name__ == "__main__":
    r = RagService().reindex()
    print(f"Réindexé : {r.received} sources -> {r.chunks_indexed} chunks.")
