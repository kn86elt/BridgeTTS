"""Internal game-mode types and public projections.

Nothing in this module is persisted by the normal chat or memory subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class GameError(Exception):
    def __init__(self, code: str, status_code: int = 409):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass
class Choice:
    id: str
    ordinal: int
    label: str
    selection: dict[str, Any]

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "number": self.ordinal, "label": self.label}


@dataclass
class Interaction:
    id: str
    kind: str
    prompt: str | None
    choices: list[Choice]
    command: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "choices": [choice.public() for choice in self.choices],
        }


@dataclass
class GameSession:
    game_id: str
    owner_page_id: str
    conversation_epoch: str
    normal_revision: int
    character_id: str
    character_name: str
    seed: dict[str, Any]
    status: str = "starting"
    revision: int = 0
    world_turn: int = 0
    memory: dict[str, Any] = field(default_factory=dict)
    scene: dict[str, Any] = field(default_factory=dict)
    pending: Interaction | None = None
    base_choices: list[Choice] = field(default_factory=list)
    public_entries: list[dict[str, Any]] = field(default_factory=list)
    request_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    operation_count: int = 0
    generation: int = 0
    started_at: str = ""
    lease_expires_at: float = 0.0
    audio_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_warning: str | None = None
