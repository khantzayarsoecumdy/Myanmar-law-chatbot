"""Human review CLI for unanswered-question events."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json

from app.feedback import REVIEWED_PATH, UNANSWERED_PATH, FEEDBACK_DIR, question_fingerprint

NON_CONTENT_FAILURES = {
    "provider_unavailable_and_no_lexical_evidence",
}


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed JSON on line {line_number}: {exc}")
    return rows


def append_jsonl(path, row):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    events = read_jsonl(UNANSWERED_PATH)
    reviewed = read_jsonl(REVIEWED_PATH)
    already_reviewed = {row.get("fingerprint") for row in reviewed}
    grouped = defaultdict(list)
    for event in events:
        fingerprint = event.get("fingerprint") or question_fingerprint(event.get("question", ""))
        grouped[fingerprint].append(event)

    pending = []
    for fingerprint, records in grouped.items():
        if fingerprint in already_reviewed:
            continue
        reasons = Counter(record.get("reason", "unknown") for record in records)
        if all(reason in NON_CONTENT_FAILURES for reason in reasons):
            continue
        newest = max(records, key=lambda row: row.get("timestamp", ""))
        pending.append((fingerprint, records, newest, reasons))
    pending.sort(key=lambda item: item[2].get("timestamp", ""))

    if not pending:
        print("No unanswered content questions are waiting for review.")
        return

    print(f"{len(pending)} distinct question(s) need review. Provider-only failures are excluded.")
    for index, (_, records, newest, reasons) in enumerate(pending, start=1):
        print(f"\n{index}. Asked {len(records)} time(s); reasons: {dict(reasons)}")
        print(f"   {newest.get('question', '')}")
    print("\nChoose a number, or press Enter to stop.")

    while True:
        choice = input("Question number: ").strip()
        if not choice:
            return
        try:
            selected = pending[int(choice) - 1]
        except (ValueError, IndexError):
            print("Enter one of the listed numbers.")
            continue

        fingerprint, records, newest, reasons = selected
        question = newest.get("question", "")
        print(f"\nQuestion: {question}")
        answer = input("Paste the exact answer verified in the document: ").strip()
        if not answer:
            print("Skipped; an answer is required.")
            continue
        source_document = input("Source document name: ").strip()
        if not source_document:
            print("Skipped; a source document is required.")
            continue
        source_quote = input("Paste the exact supporting passage: ").strip()
        if not source_quote:
            print("Skipped; a supporting passage is required.")
            continue
        verified_by = input("Reviewer name/initials (optional): ").strip()
        row = {
            "review_status": "approved",
            "fingerprint": fingerprint,
            "question": question,
            "answer": answer,
            "source_document": source_document,
            "source_quote": source_quote,
            "verified_by": verified_by,
            "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "observed_count": len(records),
            "observed_reasons": dict(reasons),
        }
        append_jsonl(REVIEWED_PATH, row)
        print(f"Saved reviewed example to {REVIEWED_PATH}")
        pending = [item for item in pending if item[0] != fingerprint]
        if not pending:
            return


if __name__ == "__main__":
    main()
