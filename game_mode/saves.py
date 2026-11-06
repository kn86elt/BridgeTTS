"""Versioned local checkpoints, separate from public chat/game logs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import Choice, GameError, GameSession, Interaction

MAX_SAVE_BYTES = 16 * 1024 * 1024


class _SaveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Choice(_SaveModel):
    id: str
    ordinal: int
    label: str
    selection: dict[str, Any]

    def restore(self) -> Choice:
        return Choice(**self.model_dump())


class _Interaction(_SaveModel):
    id: str
    kind: str
    prompt: str | None
    choices: list[_Choice]
    command: str | None

    def restore(self) -> Interaction:
        return Interaction(self.id, self.kind, self.prompt,
                           [choice.restore() for choice in self.choices], self.command)


class _State(_SaveModel):
    character_id: str
    character_name: str
    seed: dict[str, Any]
    revision: int = Field(ge=0)
    world_turn: int = Field(ge=0)
    memory: dict[str, Any]
    scene: dict[str, Any]
    pending: _Interaction | None
    base_choices: list[_Choice]
    public_entries: list[dict[str, Any]]
    operation_count: int = Field(ge=0)
    started_at: str
    choice_notice: str | None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class GameSaveStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _path(self, save_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", save_id):
            raise GameError("INVALID_SAVE_ID", 422)
        return self.directory / f"{save_id}.json"

    @staticmethod
    def identity(game_id: str, page_id: str, request_id: str) -> str:
        return _digest([game_id, page_id, request_id])

    def _read(self, save_id: str) -> dict[str, Any]:
        path = self._path(save_id)
        try:
            with path.open("rb") as source:
                raw = source.read(MAX_SAVE_BYTES + 1)
            if len(raw) > MAX_SAVE_BYTES:
                raise ValueError("save too large")
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("format") != "bridge-game-save":
                raise ValueError("invalid save format")
            if type(data.get("version")) is not int or data["version"] != 1:
                raise GameError("SAVE_VERSION_UNSUPPORTED", 422)
            if data.get("id") != save_id or data.get("checksum") != _digest(data["state"]):
                raise ValueError("invalid checksum")
            _State.model_validate(data["state"])
            if not all(isinstance(data.get(key), str) for key in ("name", "saved_at", "request_hash")):
                raise ValueError("invalid metadata")
            return data
        except FileNotFoundError as exc:
            raise GameError("SAVE_NOT_FOUND", 404) from exc
        except (ValueError, KeyError, TypeError, RecursionError, ValidationError) as exc:
            raise GameError("SAVE_CORRUPT", 422) from exc
        except OSError as exc:
            raise GameError("SAVE_READ_FAILED", 500) from exc

    @staticmethod
    def _summary(data: dict[str, Any]) -> dict[str, Any]:
        state = data["state"]
        return {"id": data["id"], "name": data["name"], "saved_at": data["saved_at"],
                "character_name": state["character_name"], "world_turn": state["world_turn"]}

    def list(self) -> dict[str, Any]:
        saves, unavailable = [], 0
        try:
            paths = list(self.directory.glob("*.json"))
        except OSError as exc:
            raise GameError("SAVE_READ_FAILED", 500) from exc
        for path in paths:
            try:
                saves.append(self._summary(self._read(path.stem)))
            except GameError:
                unavailable += 1
        saves.sort(key=lambda item: (item["saved_at"], item["id"]), reverse=True)
        return {"saves": saves, "unavailable": unavailable}

    def prior(self, save_id: str, request_hash: str) -> dict[str, Any] | None:
        try:
            data = self._read(save_id)
        except GameError as exc:
            if exc.code == "SAVE_NOT_FOUND":
                return None
            raise
        if data["request_hash"] != request_hash:
            raise GameError("REQUEST_ID_REUSED")
        return self._summary(data)

    def save(self, session: GameSession, name: str, save_id: str, request_hash: str) -> dict[str, Any]:
        state = {key: value for key, value in asdict(session).items() if key in _State.model_fields}
        data = {"format": "bridge-game-save", "version": 1, "id": save_id,
                "name": name, "saved_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "request_hash": request_hash, "checksum": _digest(state), "state": state}
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_SAVE_BYTES:
            raise GameError("SAVE_TOO_LARGE", 413)
        destination = self._path(save_id)
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            raise GameError("SAVE_WRITE_FAILED", 500) from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return self._summary(data)

    def restore(self, save_id: str, **lifecycle: Any) -> GameSession:
        saved = _State.model_validate(self._read(save_id)["state"])
        values = saved.model_dump(exclude={"pending", "base_choices"})
        return GameSession(**values, **lifecycle, status="active",
                           pending=saved.pending.restore() if saved.pending else None,
                           base_choices=[choice.restore() for choice in saved.base_choices])
