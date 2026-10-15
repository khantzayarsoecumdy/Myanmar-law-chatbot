import json
import logging
import re
import requests
from app.config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def parse_json_object(raw: str) -> dict:
    if not raw or not isinstance(raw, str):
        return {}
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _message_text(payload: dict) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _redact(value: str) -> str:
    if OPENROUTER_API_KEY:
        return value.replace(OPENROUTER_API_KEY, "[redacted]")
    return value


def _error_details(body: dict) -> tuple[str, int | None, dict]:
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        choices = body.get("choices") if isinstance(body, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        error = first.get("error") if isinstance(first, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or "Unknown OpenRouter error")
        status = error.get("status") or error.get("status_code") or error.get("code")
        metadata = error.get("metadata")
    elif error:
        message = str(error)
        status = None
        metadata = {}
    else:
        return "", None, {}

    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    metadata = metadata if isinstance(metadata, dict) else {}
    if isinstance(body, dict) and body.get("error_type") and not metadata.get("error_type"):
        metadata = {**metadata, "error_type": body["error_type"]}
    return message, status, metadata


def _should_fallback(status: int, body: dict | None = None) -> bool:
    if status in {402, 429, 503}:
        return True
    message, embedded_status, metadata = _error_details(body or {})
    if embedded_status in {402, 429, 503}:
        return True
    if metadata.get("error_type") in {
        "rate_limit_exceeded",
        "provider_overloaded",
        "provider_unavailable",
        "timeout",
        "server",
    }:
        return True
    if status == 403 and metadata.get("limit_source"):
        return True
    # OpenRouter can include an upstream error in an otherwise successful response.
    if status == 200 and message:
        return bool(re.search(
            r"rate.?limit|quota|insufficient credit|budget|provider.{0,30}(overload|unavailable)",
            message,
            re.IGNORECASE,
        ))
    return False


def _complete_ollama(messages: list[dict], timeout: int) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 1536, "repeat_penalty": 1.15},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise LLMError(504, f"Ollama timed out while using model {OLLAMA_MODEL}.") from exc
    except requests.ConnectionError as exc:
        raise LLMError(
            503,
            f"Cannot connect to Ollama at {OLLAMA_URL}. Start Ollama and make sure {OLLAMA_MODEL} is installed.",
        ) from exc

    if not response.ok:
        detail = _redact(response.text[:500] or response.reason)
        raise LLMError(503, f"Ollama returned HTTP {response.status_code}: {detail}")
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMError(502, "Ollama returned a non-JSON response.") from exc

    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMError(502, "Ollama returned no message content.")
    parsed = parse_json_object(content)
    if not parsed:
        raise LLMError(502, "Ollama did not return a valid JSON object.")
    return parsed


def _fallback_to_ollama(messages: list[dict], timeout: int, reason: str) -> dict:
    logger.warning("Falling back to Ollama because %s", reason)
    for attempt in range(2):
        try:
            return _complete_ollama(messages, timeout)
        except LLMError as exc:
            if attempt == 0 and "valid JSON object" in exc.detail:
                logger.warning("Ollama returned malformed JSON; retrying once with a strict format reminder.")
                retry_messages = [*messages, {
                    "role": "user",
                    "content": "Return only one valid JSON object matching the requested schema. Do not include prose or Markdown.",
                }]
                messages = retry_messages
                continue
            raise LLMError(
                503,
                f"{reason} Ollama fallback failed: {exc.detail}",
            ) from exc


def complete_json(messages: list[dict], timeout: int = 180) -> dict:
    if not OPENROUTER_API_KEY:
        return _fallback_to_ollama(
            messages,
            timeout,
            "OPENROUTER_API_KEY is not configured.",
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "Myanmar Law Document Chatbot",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2048,
        "reasoning": {"enabled": False, "exclude": True},
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        return _fallback_to_ollama(messages, timeout, "OpenRouter timed out.")
    except requests.ConnectionError as exc:
        return _fallback_to_ollama(messages, timeout, f"Cannot connect to OpenRouter at {OPENROUTER_URL}.")

    if response.status_code == 401:
        raise LLMError(502, "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in backend/.env.")
    body = None
    try:
        body = response.json()
    except ValueError:
        if response.status_code in {402, 429, 503}:
            return _fallback_to_ollama(
                messages,
                timeout,
                f"OpenRouter returned HTTP {response.status_code}.",
            )
        if not response.ok:
            detail = _redact(response.text[:500] or response.reason)
            raise LLMError(502, f"OpenRouter returned HTTP {response.status_code}: {detail}")
        raise LLMError(502, "OpenRouter returned a non-JSON response.")

    message, _, _ = _error_details(body) if isinstance(body, dict) else ("", None, {})
    if _should_fallback(response.status_code, body if isinstance(body, dict) else None):
        reason = f"OpenRouter reached a quota/rate limit or became unavailable (HTTP {response.status_code})."
        if message:
            reason = f"{reason} {_redact(message)}"
        return _fallback_to_ollama(messages, timeout, reason)
    if not response.ok:
        detail = _redact(message or response.text[:500] or response.reason)
        raise LLMError(502, f"OpenRouter returned HTTP {response.status_code}: {detail}")
    if message:
        raise LLMError(502, _redact(message))

    return parse_json_object(_message_text(body))
