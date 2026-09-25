from __future__ import annotations

import re
import time
import hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from unipods_rag.service import RagService
from unipods_rag.schemas import ChatMessage


# ============================================================
# CONFIGURATION
# ============================================================

CHAT_FILE = Path("chat.txt")

# IMPORTANT :
# Mets ici le JID réel du groupe WhatsApp UNIPODS.
#
# D'après les tests précédents, le groupe utilisé était :
GROUP_ID = "120363412042126507@g.us"

# Nombre de messages envoyés à RagService à la fois.
IMPORT_BATCH_SIZE = 100

# Petite pause entre les lots.
BATCH_PAUSE_SECONDS = 3


# ============================================================
# FORMAT WHATSAPP
# ============================================================

MESSAGE_RE = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}), "
    r"(\d{1,2}:\d{2}:\d{2}\s[AP]M)\] "
    r"([^:]+): (.*)$"
)

SYSTEM_RE = re.compile(
    r"^\[\d{1,2}/\d{1,2}/\d{2,4}, "
    r"\d{1,2}:\d{2}:\d{2}\s[AP]M\] - "
)


# ============================================================
# PARSING
# ============================================================

def parse_whatsapp(path: Path):
    messages = []
    current = None

    with path.open("r", encoding="utf-8") as f:

        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Ignorer les notifications système
            if SYSTEM_RE.match(line):
                continue

            match = MESSAGE_RE.match(line)

            if match:
                if current:
                    messages.append(current)

                date_part, time_part, sender, text = match.groups()

                try:
                    timestamp = datetime.strptime(
                        f"{date_part} {time_part}",
                        "%m/%d/%y %I:%M:%S %p",
                    )

                except ValueError:
                    current = None
                    continue

                current = {
                    "timestamp": timestamp,
                    "sender": sender.strip(),
                    "text": text.strip(),
                }

            elif current and line.strip():
                current["text"] += "\n" + line.strip()

        if current:
            messages.append(current)

    return messages


# ============================================================
# NETTOYAGE
# ============================================================

def clean_text(text: str) -> str:

    text = text.strip()

    # Retirer les marqueurs média tout en conservant
    # le texte accompagnant éventuellement le média.
    markers = [
        "<document omis>",
        "<image omise>",
        "<image omis>",
        "<vidéo omise>",
        "<video omis>",
        "<audio omis>",
        "<sticker omis>",
    ]

    for marker in markers:
        text = text.replace(marker, "")

    return text.strip()


def should_ignore(text: str) -> bool:

    value = text.strip().lower()

    if not value:
        return True

    ignored = {
        "ce message a été supprimé",
        "this message was deleted",
    }

    return value in ignored


# ============================================================
# MESSAGE ID
# ============================================================

def make_message_id(
    timestamp: datetime,
    sender: str,
    text: str,
) -> str:

    """
    ID déterministe.

    Relancer l'import avec le même chat.txt génère donc
    exactement les mêmes IDs.
    """

    raw = (
        f"{GROUP_ID}|"
        f"{timestamp.isoformat()}|"
        f"{sender}|"
        f"{text}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]

    return f"whatsapp-history-{digest}"


# ============================================================
# CONVERSION
# ============================================================

def build_chat_messages(parsed, timezone_name: str):

    tz = ZoneInfo(timezone_name)

    result = []

    for item in parsed:

        text = clean_text(item["text"])

        if should_ignore(text):
            continue

        timestamp = item["timestamp"]

        # L'export WhatsApp ne contient pas explicitement
        # le fuseau horaire.
        #
        # On interprète donc les heures avec le timezone
        # configuré dans le projet.
        timestamp = timestamp.replace(tzinfo=tz)

        message_id = make_message_id(
            timestamp,
            item["sender"],
            text,
        )

        message = ChatMessage(
            message_id=message_id,
            channel=GROUP_ID,
            author=item["sender"],
            text=text,
            timestamp=timestamp,
        )

        result.append(message)

    return result


# ============================================================
# IMPORT
# ============================================================

def main():

    print()
    print("========================================")
    print(" UNIPODS WhatsApp History Import")
    print("========================================")
    print()

    if not CHAT_FILE.exists():
        print(
            f"❌ Fichier introuvable : "
            f"{CHAT_FILE.resolve()}"
        )
        return

    print(f"📄 Fichier : {CHAT_FILE}")
    print(f"💬 Groupe  : {GROUP_ID}")
    print()

    # --------------------------------------------------------
    # Initialiser le RAG
    # --------------------------------------------------------

    print("Initialisation du RAG...")

    service = RagService()

    print(f"Embedder : {service.embedder.name}")
    print(f"Timezone : {service.s.timezone}")
    print()

    # --------------------------------------------------------
    # Parser
    # --------------------------------------------------------

    parsed = parse_whatsapp(CHAT_FILE)

    messages = build_chat_messages(
        parsed,
        service.s.timezone,
    )

    print(f"Messages détectés : {len(parsed)}")
    print(f"Messages à importer : {len(messages)}")
    print()

    if not messages:
        print("❌ Aucun message à importer.")
        return

    # --------------------------------------------------------
    # Confirmation visuelle
    # --------------------------------------------------------

    print("Premier message :")
    print(messages[0].timestamp)
    print(messages[0].author)
    print(messages[0].text[:200])

    print()

    print("Dernier message :")
    print(messages[-1].timestamp)
    print(messages[-1].author)
    print(messages[-1].text[:200])

    print()
    print("----------------------------------------")
    print("Début de l'import")
    print("----------------------------------------")
    print()

    # --------------------------------------------------------
    # Import par lots
    # --------------------------------------------------------

    total = len(messages)

    total_received = 0
    total_chunks = 0

    for start in range(
        0,
        total,
        IMPORT_BATCH_SIZE,
    ):

        end = min(
            start + IMPORT_BATCH_SIZE,
            total,
        )

        batch = messages[start:end]

        batch_number = (
            start // IMPORT_BATCH_SIZE
        ) + 1

        total_batches = (
            total + IMPORT_BATCH_SIZE - 1
        ) // IMPORT_BATCH_SIZE

        print(
            f"[{batch_number}/{total_batches}] "
            f"Messages {start + 1}-{end}..."
        )

        try:

            result = service.ingest_messages(batch)

            total_received += result.received
            total_chunks += result.chunks_indexed

            print(
                f"    ✅ received={result.received} "
                f"chunks={result.chunks_indexed}"
            )

        except Exception as exc:

            print()
            print("❌ Erreur pendant l'import :")
            print(exc)
            print()

            print(
                "L'import s'arrête ici. "
                "Les lots précédents restent enregistrés."
            )

            print(
                "Tu peux relancer le script : "
                "les message_id sont déterministes."
            )

            return

        if end < total:

            print(
                f"    Pause {BATCH_PAUSE_SECONDS}s..."
            )

            time.sleep(BATCH_PAUSE_SECONDS)

    # --------------------------------------------------------
    # FIN
    # --------------------------------------------------------

    print()
    print("========================================")
    print(" ✅ IMPORT TERMINÉ")
    print("========================================")
    print()

    print(
        f"Messages traités : {total_received}"
    )

    print(
        f"Chunks indexés   : {total_chunks}"
    )

    print()

    try:

        stats = service.stats()

        print("État actuel du RAG :")
        print(
            f"  Messages : {stats['messages']}"
        )
        print(
            f"  Calls    : {stats['calls']}"
        )
        print(
            f"  Chunks   : {stats['chunks']}"
        )
        print(
            f"  Embedder : {stats['embedder']}"
        )

    except Exception as exc:

        print(
            "⚠️ Impossible de lire les statistiques :",
            exc,
        )

    print()
    print(
        "L'historique WhatsApp UNIPODS "
        "est maintenant disponible pour le RAG."
    )


if __name__ == "__main__":
    main()