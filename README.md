# UniPods Group Intelligence : moteur RAG + interface (partie d'Ousmane)

Ce dossier contient le **moteur RAG** (découpage, embeddings, base vectorielle, récupération, génération)
et l'**interface conversationnelle** du bot. Il s'intègre au backend FastAPI de Gabriel en deux lignes.

## Démarrage rapide (5 minutes, sans clé ni téléchargement)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # puis mettre EMBEDDING_BACKEND=hash pour rester hors-ligne
python -m scripts.seed_demo     # charge des données fictives (dates relatives à aujourd'hui)
uvicorn unipods_rag.main:create_app --factory --reload
# ouvrir http://localhost:8000
python -m pytest                # 30 tests
```

## Passer en mode "vraie qualité"

1. `pip install -r requirements-ml.txt` puis `EMBEDDING_BACKEND=sentence-transformers` dans `.env`
   (modèle multilingue FR/EN, ~120 Mo téléchargés au premier lancement).
2. Renseigner `ANTHROPIC_API_KEY` : les réponses deviennent rédigées, avec citations [1], [2].
   Sans clé, le bot reste utile en **mode extractif** (il renvoie les passages pertinents tels quels).
3. Relancer `python -m scripts.reindex` après tout changement de modèle d'embedding.
4. Recalibrer `MIN_SCORE` et `CONFIDENT_SCORE` avec `python -m scripts.evaluate eval/questions.json`
   (les valeurs par défaut ont été réglées avec l'embedder de développement, pas avec le vrai).

## Intégration dans le backend de Gabriel

```python
from unipods_rag.service import RagService
from unipods_rag.api import create_router

rag = RagService()
app.include_router(create_router(rag))           # /rag/ingest/*, /rag/ask, /rag/digest, /rag/stats
```

Depuis le webhook de messages, il suffit d'appeler `rag.ingest_messages([...])` (ou `POST /rag/ingest/messages`).

### Contrat d'ingestion (`unipods_rag/schemas.py`)

| Objet | Champs obligatoires | Règles |
|---|---|---|
| `ChatMessage` | `message_id`, `text`, `timestamp` (+ `author`, `channel`) | `message_id` unique et stable : renvoyer le même message ne crée jamais de doublon |
| `Transcript` | `call_id`, `title`, `started_at`, `text` | une réplique par ligne, `Nom: texte`, préfixe `[hh:mm:ss]` optionnel |

Un `timestamp` sans fuseau est interprété comme UTC.

## Carte des fichiers

| Fichier | Rôle | Propriétaire |
|---|---|---|
| `schemas.py` | Contrat de données (à figer avec Gabriel) | Ousmane + Gabriel |
| `chunking.py` | Fenêtres de chat d'1 h, transcriptions par répliques | Ousmane |
| `embeddings.py`, `vector_store.py` | Embeddings et ChromaDB (autre base = une classe à écrire) | Ousmane |
| `query_understanding.py` | "hier", "l'appel", "le lien" en filtres | Ousmane |
| `retriever.py` | Sens + mots-clés + récence + bonus lien | Ousmane |
| `prompts.py` | Tous les prompts et l'assainissement du contexte | Adeyinka (+ Rosine pour la sécurité) |
| `generator.py`, `service.py` | Réponse Claude / repli extractif, orchestration | Ousmane |
| `api.py` | Routes et authentification par clé | Ousmane + Rosine |
| `frontend/index.html` | Interface FR/EN, sources dépliables, résumé | Ousmane |

## Limites connues

- Les embeddings `hash` n'ont aucune compréhension du sens : développement et tests uniquement.
- L'authentification est une clé d'API partagée. Une clé dans un navigateur n'est pas secrète : prévoir une vraie authentification par utilisateur.
- Les heures dans les extraits suivent `TIMEZONE` (serveur) ; l'interface affiche l'heure du navigateur.
- Un mot comme "call" dans une question (« call for applications ») active à tort le filtre « appels » ; l'élargissement automatique le compense.
- Les résumés sont à la demande. Un résumé périodique poussé dans le groupe demande un planificateur (cron) côté backend.
- L'audio n'est pas transcrit ici : le module reçoit du texte.
