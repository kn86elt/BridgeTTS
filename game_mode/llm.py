"""Role-separated LLM adapter for game mode."""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from .models import GameError
from .output import coerce_model_output
from .schemas import response_format

PROMPT_DIR = Path(__file__).parent / "prompts"
REQUEST_TIMEOUT_SECONDS = 180
MAX_REPAIR_ATTEMPTS = 1
JSON_TEMPERATURE = 0.3
# Generation caps per role. They bound rambling output from small models; the
# validators still enforce the character limits of the contract.
MAX_TOKENS = {"opening": 800, "resolve": 1200, "choices": 400, "summary": 2500, "character": 800}
# Try schema enforcement first, then json_object, then plain text. The first
# mode the backend accepts is remembered per client/model for the process.
FORMAT_MODES: tuple[str | None, ...] = ("json_schema", "json_object", None)
# Thinking models spend the token budget on hidden reasoning. Ask the backend
# to skip it; where that is rejected or ignored, grow the budget instead.
# Both keys are sent together: `reasoning_effort` is honoured by LM Studio with
# Gemma 4 (verified), `chat_template_kwargs.enable_thinking` by Qwen3-style
# templates on llama.cpp / vLLM / LM Studio. Backends ignore the key they do
# not understand; backends that reject unknown arguments get a retry without.
REASONING_OFF_BODY: dict[str, Any] = {
    "reasoning_effort": "none",
    "chat_template_kwargs": {"enable_thinking": False},
}
REASONING_ALLOWANCE_STEP = 1500
REASONING_ALLOWANCE_MAX = 6000

_backend_lock = threading.Lock()
_format_support: dict[str, str | None] = {}
_reasoning_effort_support: dict[str, bool] = {}
_reasoning_allowance: dict[str, int] = {}

_REASONING_OPEN = re.compile(r"<(?:think|thinking|reasoning|reflection)>", re.IGNORECASE)
_REASONING_CLOSE = re.compile(r"</(?:think|thinking|reasoning|reflection)>", re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought blocks some local models emit before the answer.

    Text after the last closing tag is the answer. An unterminated opening tag
    means the model is still thinking, so nothing after it is usable yet.
    """
    closes = list(_REASONING_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opened = _REASONING_OPEN.search(text)
    if opened:
        text = text[:opened.start()]
    return text.strip()


def _decode_json(text: str) -> Any:
    """Extract the JSON answer, tolerating reasoning blocks, fences and prose."""
    stripped = strip_reasoning(text)
    candidates = [stripped]
    candidates.extend(match.group(1).strip() for match in _FENCE.finditer(stripped))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    decoder = json.JSONDecoder()
    start = min(
        (position for opener in ("{", "[")
         if (position := stripped.find(opener)) != -1),
        default=-1,
    )
    while start != -1:
        try:
            value, _ = decoder.raw_decode(stripped, start)
        except ValueError:
            value = None
        if isinstance(value, (dict, list)):
            return value
        start = min(
            (position for opener in ("{", "[")
             if (position := stripped.find(opener, start + 1)) != -1),
            default=-1,
        )
    raise ValueError("No JSON object found in model output")


def _repeats_reply(text: str, previous: str) -> bool:
    if not previous:
        return False
    # Ignore punctuation, emoji and spacing before comparing the actual wording.
    current = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()
    earlier = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", previous)).casefold()
    if not current or not earlier:
        return text.strip() == previous.strip()
    if current == earlier:
        return True
    # Only apply fuzzy matching to long replies; short answers may differ by a negation.
    return min(len(current), len(earlier)) >= 40 and SequenceMatcher(
        None, current, earlier, autojunk=False,
    ).ratio() >= 0.95


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


_FORMAT_WORDS = ("response_format", "json_schema", "schema", "grammar")
# Error text that points at the reasoning-off arguments rather than the format.
_REASONING_ARGUMENTS = ("reasoning_effort", "chat_template_kwargs", "enable_thinking", "extra_body")


def _client_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500


def _mentions(exc: BaseException, *words: str) -> bool:
    message = str(exc).lower()
    return any(word in message for word in words)


def _rejects_response_format(exc: BaseException) -> bool:
    """True when the backend most likely refused the response_format argument."""
    return _client_error(exc) or _mentions(exc, *_FORMAT_WORDS)


class OpenAIGameLLM:
    """One request-scoped client/model pair. No normal conversation state is touched."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model or "local-model"
        self.deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
        self.on_event = None

    def set_event_sink(self, callback) -> None:
        self.on_event = callback

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def _remaining_timeout(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise GameError("LLM_TIMEOUT", 504)
        return min(REQUEST_TIMEOUT_SECONDS, remaining)

    def _format_key(self) -> str:
        return f"{getattr(self.client, 'base_url', '')}|{self.model}"

    def _create(self, role: str, messages: list[dict[str, Any]], *, stream: bool,
                temperature: float | None = None, fmt: dict[str, Any] | None = None) -> Any:
        """One chat.completions call with backend capability fallbacks.

        Returns the response. Raises the original exception for callers that
        own format fallback; reasoning_effort rejection is handled here.
        """
        key = self._format_key()
        while True:
            with _backend_lock:
                effort_ok = _reasoning_effort_support.get(key, True)
                allowance = _reasoning_allowance.get(key, 0)
            kwargs: dict[str, Any] = {
                "model": self.model, "messages": messages, "stream": stream,
                "max_tokens": MAX_TOKENS[role] + allowance, "timeout": self._remaining_timeout(),
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if fmt is not None:
                kwargs["response_format"] = fmt
            if effort_ok:
                kwargs["extra_body"] = dict(REASONING_OFF_BODY)
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if _is_timeout(exc) or not effort_ok:
                    raise
                # A format rejection is the caller's concern; only drop the
                # reasoning-off arguments when the error points at them, or when
                # there is no format argument left to blame.
                if _mentions(exc, *_REASONING_ARGUMENTS):
                    with _backend_lock:
                        _reasoning_effort_support[key] = False
                    continue
                raise

    def _grow_reasoning_allowance(self) -> bool:
        key = self._format_key()
        with _backend_lock:
            current = _reasoning_allowance.get(key, 0)
            if current >= REASONING_ALLOWANCE_MAX:
                return False
            _reasoning_allowance[key] = min(current + REASONING_ALLOWANCE_STEP, REASONING_ALLOWANCE_MAX)
        return True

    @staticmethod
    def _truncated_without_answer(response: Any, text: str) -> bool:
        choice = response.choices[0] if getattr(response, "choices", None) else None
        return getattr(choice, "finish_reason", None) == "length" and not strip_reasoning(text)

    def _complete_json(self, role: str, messages: list[dict[str, Any]]) -> str:
        key = self._format_key()
        with _backend_lock:
            mode = _format_support.get(key, FORMAT_MODES[0])
        while True:
            fmt = response_format(role, mode)
            try:
                response = self._create(role, messages, stream=False, temperature=JSON_TEMPERATURE, fmt=fmt)
            except GameError:
                raise
            except Exception as exc:
                if _is_timeout(exc):
                    raise GameError("LLM_TIMEOUT", 504) from exc
                if fmt is not None and _rejects_response_format(exc):
                    mode = FORMAT_MODES[FORMAT_MODES.index(mode) + 1]
                    with _backend_lock:
                        _format_support[key] = mode
                    continue
                raise GameError("LLM_FAILED", 502) from exc
            with _backend_lock:
                _format_support.setdefault(key, mode)
            text = (response.choices[0].message.content or "").strip()
            if self._truncated_without_answer(response, text) and self._grow_reasoning_allowance():
                # Hidden reasoning consumed the whole budget. Retry with more room
                # instead of spending the repair attempt on an empty answer.
                continue
            return text

    def _json(self, role: str, payload: dict[str, Any], validate: Callable[[dict[str, Any]], None],
              normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _read_prompt(role)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error = "invalid JSON or contract"
        raw = ""
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            self._emit({"type": "phase", "phase": role, "retry": bool(attempt)})
            if attempt:
                # Keep the original exchange so prefix caches stay warm and the
                # model sees exactly which answer failed and why.
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": json.dumps({
                        "validation_error": last_error,
                        "instruction": "Return a corrected JSON object only. Do not add facts or actions beyond the original request.",
                    }, ensure_ascii=False)},
                ]
            raw = self._complete_json(role, messages)
            try:
                value = coerce_model_output(role, _decode_json(raw))
                if not isinstance(value, dict):
                    raise ValueError("Expected an object")
                if normalize is not None:
                    value = normalize(value)
                validate(value)
                return value
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise GameError("INVALID_MODEL_OUTPUT", 502)

    def opening(self, seed: dict[str, Any]) -> dict[str, Any]:
        from .validation import validate_opening
        from .output import normalize_opening
        return self._json("opening", {"seed": seed}, validate_opening,
                          lambda value: normalize_opening(value, seed.get("character_name", "")))

    def resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .validation import validate_resolution
        from .output import normalize_resolution
        return self._json("resolve", payload, validate_resolution,
                          lambda value: normalize_resolution(value, payload.get("scene", {})))

    def choices(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The engine validates candidates individually and supplies missing ones.
        return self._json("choices", payload, lambda _value: None)

    def character_reply(self, payload: dict[str, Any]) -> str:
        messages = payload["messages"]
        previous_reply = payload.get("previous_reply", "")
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            self._emit({"type": "phase", "phase": "character", "retry": bool(attempt)})
            try:
                response = self._create("character", messages, stream=self.on_event is not None)
                if self.on_event is not None:
                    text = ""
                    try:
                        for chunk in response:
                            self._remaining_timeout()
                            if not chunk.choices:
                                continue
                            content = chunk.choices[0].delta.content
                            if isinstance(content, str):
                                text += content
                                visible = strip_reasoning(text)
                                if visible:
                                    self._emit({"type": "character_preview", "text": visible[:600]})
                                if len(visible) > 600:
                                    break
                    finally:
                        response.close()
                    text = strip_reasoning(text)
                else:
                    text = strip_reasoning(response.choices[0].message.content or "")
            except GameError:
                raise
            except Exception as exc:
                if _is_timeout(exc):
                    raise GameError("LLM_TIMEOUT", 504) from exc
                raise GameError("LLM_FAILED", 502) from exc
            invalid = (
                not text or len(text) > 600 or text.startswith("```")
                or text.lstrip().startswith(("{", "1.", "1)"))
                or text.casefold().startswith(("gm:", "進行役:", "選択肢:", "候補:"))
            )
            repeated = _repeats_reply(text, previous_reply)
            if not invalid and (not repeated or attempt == MAX_REPAIR_ATTEMPTS):
                return text
            instruction = "セリフ本文だけを1〜3文で出力してください。番号、候補、説明、JSON形式は出しません。"
            if repeated:
                instruction += (
                    "\n直前のあなた自身の返答と同じ文になっています。current_turnの相手の発話や見聞きした出来事に応じて、"
                    "新しい返答をしてください。単なる言い換えではなく、問いに答える、判断するなど会話を進めます。"
                    "知らない事実や相手の同意を作りません。\n直前の返答: " + previous_reply
                )
            messages = [
                *payload["messages"],
                {"role": "system", "content": instruction},
            ]
        raise GameError("INVALID_MODEL_OUTPUT", 502)

    def summarize(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .validation import validate_summary
        return self._json("summary", payload, validate_summary)
