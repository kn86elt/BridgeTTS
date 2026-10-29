"""Role-separated LLM adapter for game mode."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from .models import GameError

PROMPT_DIR = Path(__file__).parent / "prompts"
REQUEST_TIMEOUT_SECONDS = 180
MAX_REPAIR_ATTEMPTS = 1


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def _decode_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
        if not match:
            raise ValueError("Invalid code fence")
        stripped = match.group(1)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return value


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


class OpenAIGameLLM:
    """One request-scoped client/model pair. No normal conversation state is touched."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model or "local-model"
        self.deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS

    def _remaining_timeout(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise GameError("LLM_TIMEOUT", 504)
        return min(REQUEST_TIMEOUT_SECONDS, remaining)

    def _json(self, role: str, payload: dict[str, Any], validate: Callable[[dict[str, Any]], None],
              normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
        system = _read_prompt(role)
        user_payload: dict[str, Any] = payload
        last_error = "invalid JSON or contract"
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            if attempt:
                user_payload = {
                    "original_payload": payload,
                    "invalid_output": raw,
                    "validation_error": last_error,
                    "instruction": "Return a corrected object only. Do not add facts or actions beyond the original request.",
                }
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    stream=False,
                    timeout=self._remaining_timeout(),
                )
            except TimeoutError as exc:
                raise GameError("LLM_TIMEOUT", 504) from exc
            except Exception as exc:
                name = type(exc).__name__.lower()
                code = "LLM_TIMEOUT" if "timeout" in name else "LLM_FAILED"
                raise GameError(code, 504 if code == "LLM_TIMEOUT" else 502) from exc
            raw = (response.choices[0].message.content or "").strip()
            try:
                value = _decode_json(raw)
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
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    timeout=self._remaining_timeout(),
                )
            except TimeoutError as exc:
                raise GameError("LLM_TIMEOUT", 504) from exc
            except Exception as exc:
                name = type(exc).__name__.lower()
                code = "LLM_TIMEOUT" if "timeout" in name else "LLM_FAILED"
                raise GameError(code, 504 if code == "LLM_TIMEOUT" else 502) from exc
            text = (response.choices[0].message.content or "").strip()
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
