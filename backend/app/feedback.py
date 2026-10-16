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


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name} at line {line_number}.") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def compact_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFC", value).casefold().split())


def save_reviewed_answer(
    fingerprint: str,
    answer: str,
    source_document: str,
    source_quote: str,
    verified_by: str = "",
) -> dict:
    answer = answer.strip()
    source_document = source_document.strip()
    source_quote = source_quote.strip()
    if not all((answer, source_document, source_quote)):
        raise ValueError("အဖြေ၊ စာတမ်းအမည်နဲ့ အထောက်အထားစာပိုဒ် အားလုံးဖြည့်ပါ။")
    if compact_text(answer) not in compact_text(source_quote):
        raise ValueError("အတည်ပြုအဖြေကို ရင်းမြစ်စာပိုဒ်ထဲမှာ အတိအကျမတွေ့ပါ။ စာတမ်းနဲ့ ပြန်စစ်ပါ။")

    events = read_jsonl(UNANSWERED_PATH)
    matching = [row for row in events if row.get("fingerprint") == fingerprint]
    if not matching:
        raise ValueError("ဒီမေးခွန်းကို မဖြေနိုင်သောမေးခွန်းမှတ်တမ်းထဲမှာ မတွေ့ပါ။ စာမျက်နှာကို ပြန်ဖွင့်ပါ။")
    newest = max(matching, key=lambda row: row.get("timestamp", ""))
    reviewed = read_jsonl(REVIEWED_PATH)
    existing = next((row for row in reviewed if row.get("fingerprint") == fingerprint), None)
    if existing:
        return existing

    record = {
        "review_status": "approved",
        "fingerprint": fingerprint,
        "question": newest.get("question", ""),
        "answer": answer,
        "source_document": source_document,
        "source_quote": source_quote,
        "verified_by": verified_by.strip(),
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "observed_count": len(matching),
        "observed_reasons": sorted({row.get("reason", "unknown") for row in matching}),
    }
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, REVIEWED_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


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
