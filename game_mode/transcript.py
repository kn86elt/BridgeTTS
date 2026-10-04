"""Public-only game transcript serialization and atomic writer."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class GameTranscriptWriter:
    def __init__(self, directory: str | os.PathLike[str] = "log/game"):
        self.directory = Path(directory)
        self._lock = threading.RLock()

    @staticmethod
    def document(session: Any) -> dict[str, Any]:
        return {
            "format": "bridgetts.game.transcript",
            "version": 1,
            "game_id": session.game_id,
            "character_name": session.character_name,
            "started_at": session.started_at,
            "entries": session.public_entries,
        }

    def encode(self, session: Any) -> bytes:
        return json.dumps(self.document(session), ensure_ascii=False, indent=2).encode("utf-8")

    def save(self, session: Any) -> Path:
        payload = self.encode(session)
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            destination = self.directory / f"{session.game_id}.game.json"
            fd, temporary = tempfile.mkstemp(prefix=".game-", suffix=".tmp", dir=self.directory)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            except Exception:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
            return destination

