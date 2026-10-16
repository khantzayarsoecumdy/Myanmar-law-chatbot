"""UI-facing queue, runtime checks, and safe background training tasks."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from app.config import BASE_DIR, FINETUNE_PYTHON, PROJECT_DATA
from app.feedback import (
    FEEDBACK_DIR,
    FINETUNE_PATH,
    REVIEWED_PATH,
    UNANSWERED_PATH,
    read_jsonl,
)
from app.train_lora import MIN_TRAINING_EXAMPLES

MODEL_NAME = "myanmar-law-finetuned"
BASE_MODELS = {
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
}
TRAINING_OUTPUT_DIR = PROJECT_DATA / "feedback" / "training_output"
TASK_STATE_PATH = FEEDBACK_DIR / "training_status.json"
TASK_LOG_PATH = FEEDBACK_DIR / "training.log"
NON_CONTENT_FAILURES = {"provider_unavailable_and_no_lexical_evidence"}

_task_lock = threading.Lock()
_task_thread: threading.Thread | None = None
_runtime_cache: tuple[float, dict] | None = None


def _read_state() -> dict:
    if not TASK_STATE_PATH.exists():
        return {"state": "idle", "task": None, "message": ""}
    try:
        state = json.loads(TASK_STATE_PATH.read_text(encoding="utf-8"))
        if state.get("state") in {"running", "installing"} and not (
            _task_thread and _task_thread.is_alive()
        ):
            state.update({"state": "interrupted", "message": "Backend restarted before the task finished."})
        return state
    except (OSError, json.JSONDecodeError):
        return {"state": "idle", "task": None, "message": ""}


def _write_state(state: dict) -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TASK_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(TASK_STATE_PATH)


def _read_log_tail(max_bytes: int = 12000) -> str:
    if not TASK_LOG_PATH.exists():
        return ""
    try:
        with TASK_LOG_PATH.open("rb") as handle:
            handle.seek(max(0, TASK_LOG_PATH.stat().st_size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def training_runtime() -> dict:
    """Check the selected Python environment without loading the model weights."""
    global _runtime_cache
    now = time.monotonic()
    if _runtime_cache and now - _runtime_cache[0] < 30:
        return dict(_runtime_cache[1])

    configured = FINETUNE_PYTHON or sys.executable
    executable = shutil.which(configured) or (configured if Path(configured).is_file() else None)
    result = {
        "python": executable or configured,
        "dependencies_available": False,
        "cuda_available": False,
        "missing_dependencies": [],
        "ready": False,
        "message": "",
    }
    if not executable:
        result["message"] = "FINETUNE_PYTHON ကို training သုံးမည့် Python လမ်းကြောင်းအဖြစ် သတ်မှတ်ပါ။"
    else:
        # Keep this small preflight separate from the API process and never download model files here.
        check_code = (
            "import importlib.util,json; names=['torch','transformers','datasets','peft','trl']; "
            "missing=[name for name in names if importlib.util.find_spec(name) is None]; "
            "cuda=False; "
            "exec('import torch\\ncuda=torch.cuda.is_available()' if importlib.util.find_spec('torch') else ''); "
            "print(json.dumps({'missing':missing,'cuda':cuda}))"
        )
        try:
            checked = subprocess.run(
                [executable, "-c", check_code],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=35,
                check=False,
            )
            if checked.returncode == 0:
                payload = json.loads(checked.stdout.strip().splitlines()[-1])
                result["missing_dependencies"] = payload.get("missing", [])
                result["dependencies_available"] = not result["missing_dependencies"]
                result["cuda_available"] = bool(payload.get("cuda"))
                result["ready"] = result["dependencies_available"] and result["cuda_available"]
            else:
                result["message"] = checked.stderr.strip()[-1200:] or "Training Python preflight failed."
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            result["message"] = f"Training Python preflight failed: {type(exc).__name__}."

        if not result["ready"] and not result["message"]:
            if result["missing_dependencies"]:
                result["message"] = "လိုအပ်သော training packages များ မတပ်ဆင်ရသေးပါ။"
            elif not result["cuda_available"]:
                result["message"] = "ရွေးထားသော Python ပတ်ဝန်းကျင်တွင် CUDA GPU မတွေ့ပါ။"
    _runtime_cache = (now, result)
    return dict(result)


def fine_tuning_status() -> dict:
    try:
        events = read_jsonl(UNANSWERED_PATH)
        reviewed = read_jsonl(REVIEWED_PATH)
        training_examples = read_jsonl(FINETUNE_PATH)
    except ValueError as exc:
        return {
            "error": str(exc),
            "pending_questions": [],
            "reviewed_count": 0,
            "training_examples": 0,
            "runtime": training_runtime(),
            "task": _read_state(),
            "log": _read_log_tail(),
        }

    reviewed_fingerprints = {row.get("fingerprint") for row in reviewed}
    grouped = defaultdict(list)
    for event in events:
        fingerprint = event.get("fingerprint")
        if fingerprint:
            grouped[fingerprint].append(event)

    pending = []
    provider_only_count = 0
    for fingerprint, records in grouped.items():
        if fingerprint in reviewed_fingerprints:
            continue
        reasons = Counter(row.get("reason", "unknown") for row in records)
        if all(reason in NON_CONTENT_FAILURES for reason in reasons):
            provider_only_count += 1
            continue
        newest = max(records, key=lambda row: row.get("timestamp", ""))
        oldest = min(records, key=lambda row: row.get("timestamp", ""))
        pending.append({
            "fingerprint": fingerprint,
            "question": newest.get("question", ""),
            "attempts": len(records),
            "first_seen": oldest.get("timestamp"),
            "last_seen": newest.get("timestamp"),
            "reasons": dict(reasons),
        })
    pending.sort(key=lambda row: row.get("last_seen") or "", reverse=True)
    current_model = "qwen2.5:7b"
    try:
        from app import llm
        current_model = llm.OLLAMA_MODEL
    except Exception:
        pass

    return {
        "pending_questions": pending,
        "pending_count": len(pending),
        "provider_only_count": provider_only_count,
        "reviewed_count": len(reviewed),
        "training_examples": len(training_examples),
        "minimum_examples": MIN_TRAINING_EXAMPLES,
        "runtime": training_runtime(),
        "task": _read_state(),
        "log": _read_log_tail(),
        "ollama_available": shutil.which("ollama") is not None,
        "ollama_model": current_model,
        "trained_model_name": MODEL_NAME,
        "model_trained": (TRAINING_OUTPUT_DIR / "merged").is_dir(),
    }


def _start_task(task: str, base_model: str = "", epochs: float = 3) -> dict:
    global _task_thread
    with _task_lock:
        current = _read_state()
        if _task_thread and _task_thread.is_alive():
            raise RuntimeError("A fine-tuning task is already running. Wait for it to finish.")
        if current.get("state") in {"running", "installing"}:
            raise RuntimeError("ယခင်လုပ်ငန်းစဉ် ပြီးဆုံးကြောင်း မသိရသေးပါ။ အခြေအနေကို ပြန်စစ်ပါ။")

        initial_state = {
            "state": "running" if task == "training" else "installing",
            "task": task,
            "base_model": base_model or None,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None,
            "message": "",
            "return_code": None,
        }
        _write_state(initial_state)
        _task_thread = threading.Thread(
            target=_run_task,
            args=(task, base_model, epochs, initial_state),
            daemon=True,
        )
        _task_thread.start()
        return initial_state


def start_training(base_model: str, epochs: float = 3) -> dict:
    if base_model not in BASE_MODELS:
        raise ValueError("ရွေးချယ်ထားသော base model ကို ခွင့်မပြုပါ။")
    runtime = training_runtime()
    if not runtime["ready"]:
        raise RuntimeError(runtime["message"] or "Training environment အဆင်သင့်မဖြစ်သေးပါ။")
    examples = read_jsonl(FINETUNE_PATH)
    if len(examples) < MIN_TRAINING_EXAMPLES:
        raise RuntimeError(f"Training အတွက် အတည်ပြုပြီးသား ဥပမာ {MIN_TRAINING_EXAMPLES} ခု အနည်းဆုံးလိုပါတယ်။")
    return _start_task("training", base_model, epochs)


def start_ollama_install() -> dict:
    if shutil.which("ollama") is None:
        raise RuntimeError("ဒီစနစ်မှာ Ollama command ကို မတွေ့ပါ။ Ollama ထည့်သွင်းပြီး backend ကို ပြန်စပါ။")
    modelfile = TRAINING_OUTPUT_DIR / "Modelfile"
    if not modelfile.is_file():
        raise RuntimeError("Fine-tuned model မတွေ့ပါ။ ပထမဦးစွာ training ပြီးအောင်လုပ်ပါ။")
    return _start_task("install")


def _run_task(task: str, base_model: str, epochs: float, initial_state: dict) -> None:
    executable = training_runtime().get("python")
    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        with TASK_LOG_PATH.open("w", encoding="utf-8", newline="\n") as log_file:
            if task == "training":
                command = [
                    executable,
                    "-u",
                    "-m",
                    "app.train_lora",
                    "--base-model",
                    base_model,
                    "--epochs",
                    str(epochs),
                    "--output-dir",
                    str(TRAINING_OUTPUT_DIR),
                ]
            else:
                command = [
                    shutil.which("ollama") or "ollama",
                    "create",
                    MODEL_NAME,
                    "-f",
                    str(TRAINING_OUTPUT_DIR / "Modelfile"),
                ]
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return_code = process.wait()
        complete_state = {
            **initial_state,
            "state": ("trained" if task == "training" else "installed") if return_code == 0 else "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "return_code": return_code,
            "message": "" if return_code == 0 else "လုပ်ငန်းစဉ် မအောင်မြင်ပါ။ အောက်က log ကိုဖတ်ပါ။",
        }
    except Exception as exc:
        complete_state = {
            **initial_state,
            "state": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "message": f"လုပ်ငန်းစဉ် မအောင်မြင်ပါ: {type(exc).__name__}: {exc}",
        }
    try:
        _write_state(complete_state)
    except OSError:
        return


def activate_trained_model() -> dict:
    ollama = shutil.which("ollama")
    if not ollama:
        raise RuntimeError("Ollama command ကို မတွေ့ပါ။")
    try:
        shown = subprocess.run(
            [ollama, "show", MODEL_NAME],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Ollama model ကို စစ်မရပါ: {type(exc).__name__}။") from exc
    if shown.returncode != 0:
        raise RuntimeError("Ollama ထဲမှာ fine-tuned model မတွေ့ပါ။ 'ထည့်သွင်းရန်' ကို အရင်နှိပ်ပါ။")

    env_path = BASE_DIR / ".env"
    old_content = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = old_content.splitlines()
    updated = False
    for index, line in enumerate(lines):
        if re.match(r"^\s*OLLAMA_MODEL\s*=", line):
            lines[index] = f"OLLAMA_MODEL={MODEL_NAME}"
            updated = True
    if not updated:
        lines.append(f"OLLAMA_MODEL={MODEL_NAME}")
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    os.environ["OLLAMA_MODEL"] = MODEL_NAME
    from app import config, llm
    config.OLLAMA_MODEL = MODEL_NAME
    llm.OLLAMA_MODEL = MODEL_NAME

    state = _read_state()
    state["fallback_active"] = True
    state["message"] = "Fine-tuned model ကို Ollama fallback အဖြစ် အသုံးပြုနေပါပြီ။"
    _write_state(state)
    return {"model": MODEL_NAME, "message": state["message"]}
