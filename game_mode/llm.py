"""Role-separated LLM adapter for game mode."""

from __future__ import annotations

import json
import re
import time
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

    def _json(self, role: str, payload: dict[str, Any], validate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
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
                validate(value)
                return value
            except Exception as exc:
                last_error = type(exc).__name__
        raise GameError("INVALID_MODEL_OUTPUT", 502)

    def opening(self, seed: dict[str, Any]) -> dict[str, Any]:
        from .validation import validate_opening
        return self._json("opening", {"seed": seed}, validate_opening)

    def resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .validation import validate_resolution
        return self._json("resolve", payload, validate_resolution)

    def choices(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .validation import validate_choices
        return self._json("choices", payload, validate_choices)

    def character_reply(self, payload: dict[str, Any]) -> str:
        messages = payload["messages"]
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
            if not invalid:
                return text
            messages = [
                *payload["messages"],
                {"role": "system", "content": "セリフ本文だけを1〜3文で出力してください。番号、候補、説明、JSON形式は出しません。"},
            ]
        raise GameError("INVALID_MODEL_OUTPUT", 502)

    def summarize(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .validation import validate_summary
        return self._json("summary", payload, validate_summary)
