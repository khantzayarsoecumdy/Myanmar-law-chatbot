"""Local, append-only collection of questions the document QA path could not answer."""

from datetime import datetime, timezone
import hashlib
import json
import threading
import unicodedata

from app.config import PROJECT_DATA

FEEDBACK_DIR = PROJECT_DATA / "feedback"
UNANSWERED_PATH = FEEDBACK_DIR / "unanswered.jsonl"
REVIEWED_PATH = FEEDBACK_DIR / "reviewed_qa.jsonl"
FINETUNE_PATH = FEEDBACK_DIR / "finetune.jsonl"
_WRITE_LOCK = threading.Lock()


def question_fingerprint(question: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", question).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def log_unanswered(question: str, reason: str) -> None:
    """Append one local event; a logging failure must not fail the answer request."""
    cleaned = " ".join(question.split())
    if not cleaned:
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": question_fingerprint(cleaned),
        "question": cleaned,
        "reason": reason,
    }
    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, UNANSWERED_PATH.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        # Feedback logging is best effort and must never mask a valid API response.
        return
