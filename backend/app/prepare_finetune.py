"""Export only human-approved, source-grounded Q&A into chat fine-tuning JSONL."""

import json
import unicodedata

from app.feedback import FINETUNE_PATH, FEEDBACK_DIR, REVIEWED_PATH

SYSTEM_PROMPT = (
    "သင်သည် ထည့်သွင်းထားသော မြန်မာဥပဒေစာတမ်းများကိုသာ အခြေခံဖြေဆိုသော အကူအညီပေးစနစ်ဖြစ်သည်။ "
    "မေးခွန်းနှင့် တိုက်ရိုက်သက်ဆိုင်သည့် ပေးထားသောစာပိုဒ်မှသာ အဖြေကို ကောက်နုတ်ပါ။ "
    "စာပိုဒ်ထဲတွင် အဖြေမရှိလျှင် မရှိကြောင်းပြောပါ။ မခန့်မှန်းပါနှင့်၊ စာတမ်းပြင်ပဥပဒေအချက်အလက် မဖြည့်ပါနှင့်။"
)


def compact(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).casefold().split())


def read_rows():
    if not REVIEWED_PATH.exists():
        return []
    rows = []
    with REVIEWED_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed reviewed_qa.jsonl at line {line_number}: {exc}") from exc
    return rows


def make_user_message(row: dict) -> str:
    return (
        f"မေးခွန်း:\n{row['question'].strip()}\n\n"
        f"ရင်းမြစ်စာတမ်း: {row['source_document'].strip()}\n"
        f"ထည့်သွင်းထားသော စာပိုဒ်:\n{row['source_quote'].strip()}\n\n"
        "စာပိုဒ်မှ တိုက်ရိုက်ဖြေဆိုနိုင်သည့် အကြောင်းအရာကိုသာ ကောက်နုတ်ပါ။"
    )


def main():
    rows = read_rows()
    messages = []
    seen = set()
    skipped = 0
    for row in rows:
        if row.get("review_status") != "approved":
            skipped += 1
            continue
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        source_document = row.get("source_document", "").strip()
        source_quote = row.get("source_quote", "").strip()
        if not all((question, answer, source_document, source_quote)):
            skipped += 1
            continue
        if compact(answer) not in compact(source_quote):
            print(f"Skipping unsupported answer for: {question}")
            skipped += 1
            continue
        key = compact(question)
        if key in seen:
            print(f"Skipping duplicate question: {question}")
            skipped += 1
            continue
        seen.add(key)
        messages.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(row)},
                {"role": "assistant", "content": answer},
            ]
        })

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with FINETUNE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in messages:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(messages)} source-checked example(s); skipped {skipped} row(s).")
    print(f"Training file: {FINETUNE_PATH}")
    if len(messages) < 10:
        print("Collect more reviewed examples before training; the trainer requires 10 by default.")


if __name__ == "__main__":
    main()
