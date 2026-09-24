"""Charge des données FICTIVES (dates relatives à aujourd'hui) pour tester le bot sans le backend de Gabriel.

    python -m scripts.seed_demo

Les dates étant relatives, "yesterday's call" et "today" fonctionnent quel que soit le jour où on lance le script.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from unipods_rag.config import Settings
from unipods_rag.schemas import ChatMessage, Transcript
from unipods_rag.service import RagService


def build_demo(tz: ZoneInfo) -> tuple[list[ChatMessage], list[Transcript]]:
    """Données fictives, dates relatives à aujourd'hui."""
    today = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)

    def at(days_ago: int, hh: int, mm: int = 0) -> datetime:
        return (today - timedelta(days=days_ago)).replace(hour=hh, minute=mm).astimezone(timezone.utc)

    rows = [
        (2, 9, 5, "Amina", "Reminder: the hackathon team declaration is due Thursday, close of business."),
        (2, 9, 12, "Kwame", "Does anyone have the pitch deck template?"),
        (2, 9, 20, "Thandi", "Here it is: https://example.org/unipods/pitch-template (duplicate it before editing)"),
        (2, 15, 40, "Jean", "Mentors are available for 20-minute slots this week. Book one if you are stuck on your data pipeline."),
        (1, 10, 2, "Chidi", "The submission link is live: https://example.org/unipods/submit. Deadline is Thursday 24th, 23:59 GMT."),
        (1, 10, 6, "Amina", "Thanks Chidi! Remember the deliverables: working chatbot, repo, and a README."),
        (1, 11, 30, "Jean", "N'oubliez pas de remplir le formulaire de présence: https://example.org/unipods/attendance"),
        (1, 12, 15, "Kwame", "Is anyone using Pinecone or is everyone on ChromaDB?"),
        (1, 12, 18, "Thandi", "We went with ChromaDB, it runs locally and needs no account."),
        (1, 17, 45, "Chidi", "Quick poll: should the demo be in English or French? Reply with a thumbs up for English."),
        (0, 8, 10, "Amina", "Good morning all. Demo day rehearsal is at 5pm today, please join on time."),
        (0, 8, 25, "Kwame", "Rehearsal noted. I will share the dataset access link before noon: https://example.org/unipods/dataset"),
        (0, 9, 2, "Thandi", "Has anyone had trouble with the embeddings model download on slow connections?"),
        (0, 9, 8, "Jean", "Yes. Use the smaller multilingual model, it is about 120 MB and works well for French and English."),
    ]
    messages = [
        ChatMessage(message_id=f"demo-{i}", channel="general", author=a, text=txt, timestamp=at(d, h, m))
        for i, (d, h, m, a, txt) in enumerate(rows)
    ]
    messages.append(ChatMessage(message_id="demo-ann", channel="announcements", author="Organizers",
                                text="Demo day is Friday 25th at 14:00 GMT. Each team gets 5 minutes plus 3 minutes of questions.",
                                timestamp=at(1, 16, 0)))

    weekly = Transcript(
        call_id="demo-call-weekly", title="Weekly cohort call", started_at=at(1, 15, 0),
        text=(
            "[00:00:20] Amina: Welcome everyone. Let's go through the action items from this week.\n"
            "[00:02:10] Amina: Action item one: Kwame will share the dataset access link by Friday noon.\n"
            "[00:03:05] Kwame: Confirmed, I will post it in the group and email the mentors as well.\n"
            "[00:05:40] Jean: Action item two: I will book the demo day room and confirm the mentor schedule by Thursday.\n"
            "[00:08:15] Thandi: On the vector database, we decided on ChromaDB for the hackathon. Pinecone can come later if we need to scale.\n"
            "[00:11:00] Chidi: Deadline reminder: the submission closes Thursday at 23:59 GMT, and late entries will not be reviewed.\n"
            "[00:13:30] Amina: Last item: everyone must fill the attendance form so the organizers can count teams.\n"
        ),
    )
    mentor = Transcript(
        call_id="demo-call-mentor", title="Mentor Q&A on evaluating RAG", started_at=at(6, 14, 0),
        text=(
            "Mentor: To evaluate a retrieval system, build a small gold set of questions with the sources that should be returned.\n"
            "Mentor: Measure hit rate at k and mean reciprocal rank before you touch the prompts. Fix retrieval first, then generation.\n"
            "Amina: What chunk size do you recommend for chat data?\n"
            "Mentor: Group messages by time window rather than by fixed character count, so a question and its answer stay together.\n"
            "Mentor: Always show sources with the answer, people trust the bot more when they can verify it.\n"
        ),
    )
    return messages, [weekly, mentor]


def main() -> None:
    settings = Settings.from_env()
    service = RagService(settings)
    messages, transcripts = build_demo(ZoneInfo(settings.timezone))
    results = [service.ingest_messages(messages)] + [service.ingest_transcript(t) for t in transcripts]
    print(f"Démo chargée : {len(messages)} messages, {len(transcripts)} appels -> {sum(r.chunks_indexed for r in results)} chunks.")
    print(service.stats())


if __name__ == "__main__":
    main()
