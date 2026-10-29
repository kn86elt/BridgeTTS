"""Transactional, in-memory game-mode engine.

LLM work is done against a deep copy outside locks. A result is published only
if the owner, revision, and lifecycle generation still match at commit time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from .memory import add_turn, compact_if_needed, new_memory, view_context
from .models import Choice, GameError, GameSession, Interaction
from .validation import MAX_CHOICE_TEXT_CHARS
from .output import normalize_opening, normalize_resolution, prepare_choices

LEASE_SECONDS = 30 * 60
MAX_INPUT_CHARS = 2_000
MAX_WORLD_TURNS = 500
MAX_OPERATIONS = 2_000
MAX_SESSION_BYTES = 5 * 1024 * 1024
HOOK_INTERVAL_TURNS = 3
TOMBSTONE_SECONDS = 5 * 60


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


class GameEngine:
    def __init__(
        self,
        llm: Any,
        *,
        gate_lock: threading.RLock | None = None,
        normal_busy: Callable[[], bool] | None = None,
        conversation_state: Callable[[], tuple[str, int]] | None = None,
        transcript_save: Callable[[GameSession], str | None] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.llm = llm
        self.gate_lock = gate_lock or threading.RLock()
        self.normal_busy = normal_busy or (lambda: False)
        self.conversation_state = conversation_state or (lambda: ("", 0))
        self.transcript_save = transcript_save
        self.clock = clock
        self._lock = threading.RLock()
        self._session: GameSession | None = None
        self._closed_pages: dict[str, float] = {}
        self._closed_games: dict[str, tuple[str, float]] = {}
        # Start requests exist before a game_id can be returned. Keep their
        # idempotency records separately so a failed/lost start is never run twice.
        self._start_records: dict[tuple[str, str], dict[str, Any]] = {}

    def _snapshot_llm(self, on_event=None) -> Any:
        llm = self.llm.snapshot() if hasattr(self.llm, "snapshot") else self.llm
        if hasattr(llm, "set_event_sink"):
            llm.set_event_sink(on_event)
        return llm

    def _prune_locked(self) -> None:
        now = self.clock()
        for mapping in (self._closed_pages,):
            for key, expiry in list(mapping.items()):
                if expiry <= now:
                    mapping.pop(key, None)
        for game_id, (_, expiry) in list(self._closed_games.items()):
            if expiry <= now:
                self._closed_games.pop(game_id, None)
        for key, record in list(self._start_records.items()):
            if record.get("expires_at", 0) <= now:
                self._start_records.pop(key, None)
        session = self._session
        if session and session.lease_expires_at <= now:
            self._close_locked(session, "lease_expired")

    def cleanup_expired(self) -> None:
        with self._lock:
            self._prune_locked()

    def _guard_owner(self, session: GameSession, page_id: str, epoch: str | None = None) -> None:
        if session.owner_page_id != page_id:
            raise GameError("OWNER_MISMATCH")
        if epoch is not None and session.conversation_epoch != epoch:
            raise GameError("STALE_CONVERSATION")
        if session.lease_expires_at <= self.clock():
            self._close_locked(session, "lease_expired")
            raise GameError("GAME_CLOSED", 410)

    def guard_normal(self, page_id: str | None, epoch: str | None) -> None:
        """Called under gate_lock before a normal request reserves generation."""
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session:
                return
            if session.status == "closed":
                return
            if page_id and session.owner_page_id != page_id:
                raise GameError("OWNER_MISMATCH")
            if epoch and session.conversation_epoch != epoch:
                raise GameError("STALE_CONVERSATION")
            if session.status != "suspended":
                raise GameError("GAME_BUSY")
            if not page_id or session.owner_page_id != page_id:
                raise GameError("OWNER_MISMATCH")

    def blocks_direct_audio(self, page_id: str | None = None) -> None:
        with self._lock:
            self._prune_locked()
            session = self._session
            if session and session.status in {"starting", "active", "resolving"}:
                if page_id and page_id != session.owner_page_id:
                    raise GameError("OWNER_MISMATCH")
                raise GameError("GAME_BUSY")

    @staticmethod
    def _record(session: GameSession, request_id: str, payload: Any) -> tuple[str, dict[str, Any] | None]:
        if not request_id or len(request_id) > 128:
            raise GameError("INVALID_REQUEST", 422)
        digest = _stable_hash(payload)
        record = session.request_records.get(request_id)
        if record:
            if record["payload_hash"] != digest:
                raise GameError("REQUEST_ID_REUSED")
            status = record["status"]
            if status == "running":
                raise GameError("REQUEST_RUNNING", 202)
            if status == "failed":
                raise GameError(record.get("error", "REQUEST_FAILED"), record.get("http_status", 502))
            return "committed", GameEngine._response_from_record(session, record)
        session.request_records[request_id] = {
            "payload_hash": digest,
            "status": "running",
            "started_at": GameEngine._now_for_record(),
        }
        session.operation_count += 1
        return "new", None

    @staticmethod
    def _now_for_record() -> str:
        return _now_iso()

    @staticmethod
    def _finish_record(session: GameSession, request_id: str, response: dict[str, Any]) -> None:
        fields = {key: copy.deepcopy(value) for key, value in response.items() if key not in {"events", "transcript"}}
        event_ids = [event.get("id") for event in response.get("events", []) if event.get("id")]
        session.request_records[request_id].update({
            "status": "committed",
            "response": {
                "_event_ids": event_ids,
                "_transcript_ids": [event.get("id") for event in response.get("transcript", []) if event.get("id")],
                "_fields": fields,
            },
        })

    @staticmethod
    def _response_from_record(session: GameSession, record: dict[str, Any]) -> dict[str, Any]:
        response = record.get("response", {})
        if "_event_ids" not in response:
            return copy.deepcopy(response)
        wanted = set(response["_event_ids"])
        result = copy.deepcopy(response.get("_fields", {}))
        result["events"] = [copy.deepcopy(event) for event in session.public_entries if event.get("id") in wanted]
        transcript_ids = set(response.get("_transcript_ids", []))
        if transcript_ids:
            result["transcript"] = [copy.deepcopy(event) for event in session.public_entries if event.get("id") in transcript_ids]
        return result

    @staticmethod
    def _fail_record(session: GameSession, request_id: str, error: GameError) -> None:
        record = session.request_records.get(request_id)
        if record:
            record.update({"status": "failed", "error": error.code, "http_status": error.status_code})

    def _validate_limits(self, session: GameSession) -> None:
        if session.operation_count >= MAX_OPERATIONS:
            raise GameError("SESSION_LIMIT", 413)
        if session.world_turn >= MAX_WORLD_TURNS:
            raise GameError("SESSION_LIMIT", 413)

    @staticmethod
    def _add_public(
        session: GameSession,
        kind: str,
        text: str | None = None,
        choices: list[Choice] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": _id("event"),
            "revision": session.revision + 1,
            "time": _now_iso(),
            "type": kind,
        }
        if text is not None:
            entry["text"] = text
        if choices is not None:
            entry["choices"] = [choice.public() for choice in choices]
        session.public_entries.append(entry)
        return entry

    @staticmethod
    def _world_event(
        session: GameSession,
        kind: str,
        text_by_view: dict[str, str],
        *,
        source: str,
        actor_id: str | None = None,
        target_ids: list[str] | None = None,
        outcome: str = "occurred",
        important: bool = False,
        world_turn: int | None = None,
    ) -> dict[str, Any]:
        return {
            "id": _id("we"),
            "world_turn": session.world_turn if world_turn is None else world_turn,
            "source": source,
            "kind": kind,
            "actor_id": actor_id,
            "target_ids": list(target_ids or []),
            "text_by_view": {view: text for view, text in text_by_view.items() if text},
            "outcome": outcome,
            "important": important,
        }

    @staticmethod
    def _new_choices(llm_output: dict[str, Any], scene: dict[str, Any], expected_kind: str | None = None) -> list[Choice]:
        entities = {e["key"]: e for e in scene.get("entities", [])}
        choices: list[Choice] = []
        for ordinal, candidate in enumerate(llm_output["choices"], 1):
            steps: list[dict[str, Any]] = []
            labels: list[str] = []
            for raw_step in candidate["steps"]:
                kind = raw_step["kind"]
                target_key = raw_step.get("target_key") or None
                target_id = None
                if target_key:
                    entity = entities.get(target_key)
                    if not entity or "player" not in entity.get("perceived_by", []):
                        raise GameError("INVALID_MODEL_OUTPUT", 502)
                    target_id = entity["id"]
                text = raw_step["text"].strip()
                if not text or len(text) > MAX_CHOICE_TEXT_CHARS:
                    raise GameError("INVALID_MODEL_OUTPUT", 502)
                steps.append({"kind": kind, "text": text, "target_id": target_id})
                labels.append(f"「{text}」" if kind == "speech" else text)
            label = "、".join(labels)
            if len(label) > MAX_CHOICE_TEXT_CHARS + 2 or len(steps) != 1:
                raise GameError("INVALID_MODEL_OUTPUT", 502)
            if expected_kind == "content" and any(step["kind"] != "speech" for step in steps):
                raise GameError("INVALID_MODEL_OUTPUT", 502)
            choices.append(Choice(_id("choice"), ordinal, label, {"kind": "perform", "steps": steps}))
        if len(choices) != 3:
            raise GameError("INVALID_MODEL_OUTPUT", 502)
        return choices

    @staticmethod
    def _make_interaction(kind: str, prompt: str | None, choices: list[Choice], command: str | None = None) -> Interaction:
        return Interaction(_id("interaction"), kind, prompt, choices, command)

    @staticmethod
    def _player_context(session: GameSession) -> dict[str, Any]:
        return view_context(session.memory, "player", include_entity_keys=True)

    def _character_reply(self, session: GameSession, llm: Any) -> str:
        context = view_context(session.memory, "character")
        context["current_turn"] = next(
            (copy.deepcopy(turn) for turn in context["recent_turns"] if turn["world_turn"] == session.world_turn),
            {"world_turn": session.world_turn, "events": []},
        )
        normal_history = [
            {"role": item.get("role"), "content": item.get("content", "")}
            for item in session.seed.get("normal_history", [])
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        persona = session.seed.get("system_prompt", "")
        messages = [{"role": "system", "content": persona + "\n\n" + self._character_instruction()}]
        messages.extend(normal_history[-24:])
        messages.append({"role": "user", "content": json.dumps(context, ensure_ascii=False)})
        previous_reply = next((entry["text"] for entry in reversed(session.public_entries)
                               if entry["type"] == "character"), "")
        return llm.character_reply({"messages": messages, "previous_reply": previous_reply})

    @staticmethod
    def _character_instruction() -> str:
        from pathlib import Path
        return (Path(__file__).parent / "prompts" / "character_context.txt").read_text(encoding="utf-8")

    def _choices(self, session: GameSession, llm: Any, character_text: str, kind: str | None = None) -> list[Choice]:
        try:
            output = llm.choices({
                "kind": kind or ("reply" if character_text else "action"),
                "player_context": self._player_context(session),
                "last_character_text": character_text,
                "character_name": session.character_name,
            })
        except Exception:
            # A completed world turn must not be rolled back for optional suggestions.
            output = {}
        output, supplemented = prepare_choices(output, session.scene, content_only=kind == "content")
        session.choice_notice = "候補を一部補いました。自由入力でも行動や発話を指定できます。" if supplemented else None
        from .validation import validate_choices
        validate_choices(output)
        return self._new_choices(output, session.scene, expected_kind="content" if kind == "content" else None)

    def _install_choices(self, session: GameSession, choices: list[Choice], character_text: str = "") -> None:
        session.base_choices = copy.deepcopy(choices)
        session.pending = self._make_interaction("reply" if character_text else "action", None, choices)

    def _view(
        self,
        session: GameSession,
        events: list[dict[str, Any]] | None = None,
        *,
        include_transcript: bool = True,
    ) -> dict[str, Any]:
        view = {
            "game_id": session.game_id,
            "request_id": None,
            "status": session.status,
            "revision": session.revision,
            "world_turn": session.world_turn,
            "events": copy.deepcopy(events if events is not None else session.public_entries),
            "interaction": session.pending.public() if session.pending else None,
            "log_warning": session.log_warning,
            "notice": session.choice_notice,
        }
        if include_transcript:
            view["transcript"] = copy.deepcopy(session.public_entries)
        return view

    def _persist(self, session: GameSession) -> None:
        if not self.transcript_save:
            return
        try:
            session.log_warning = self.transcript_save(session) or None
        except Exception:
            session.log_warning = "公開ログを自動保存できませんでした。記録はこの画面中のRAMに保持しています。"

    def _check_size(self, session: GameSession) -> None:
        serial = json.dumps({
            "scene": session.scene,
            "memory": session.memory,
            "entries": session.public_entries,
            "requests": session.request_records,
            "audio": session.audio_records,
        }, ensure_ascii=False)
        if len(serial.encode("utf-8")) > MAX_SESSION_BYTES:
            raise GameError("SESSION_LIMIT", 413)

    def _close_locked(self, session: GameSession, reason: str) -> None:
        if self._session is not session:
            return
        session.status = "closed"
        session.generation += 1
        self._fail_running_records(session, "GAME_CLOSED", 410)
        for start_record in self._start_records.values():
            if start_record.get("game_id") == session.game_id:
                start_record["expires_at"] = self.clock() + TOMBSTONE_SECONDS
        self._persist(session)
        self._closed_games[session.game_id] = (session.owner_page_id, self.clock() + TOMBSTONE_SECONDS)
        # Keep only protocol tombstones; the seed, scene, and game memory are discarded.
        self._session = None

    def _fail_running_records(self, session: GameSession, code: str, status_code: int) -> None:
        for request_id, record in session.request_records.items():
            if record.get("status") == "running":
                record.update({"status": "failed", "error": code, "http_status": status_code})
                start_record = self._start_records.get((session.owner_page_id, request_id))
                if start_record and start_record.get("status") == "running":
                    start_record.update({"status": "failed", "error": code, "http_status": status_code})

    def start(
        self,
        *,
        page_id: str,
        conversation_epoch: str,
        normal_revision: int,
        request_id: str,
        seed: dict[str, Any] | Callable[[], dict[str, Any]],
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not page_id or len(page_id) > 128 or not conversation_epoch:
            raise GameError("INVALID_REQUEST", 422)
        if not request_id or len(request_id) > 128:
            raise GameError("INVALID_REQUEST", 422)
        payload_hash = _stable_hash({"page_id": page_id, "epoch": conversation_epoch, "normal_revision": normal_revision})
        with self.gate_lock:
            with self._lock:
                self._prune_locked()
                key = (page_id, request_id)
                prior = self._start_records.get(key)
                if prior:
                    if prior["payload_hash"] != payload_hash:
                        raise GameError("REQUEST_ID_REUSED")
                    if prior["status"] == "committed":
                        return copy.deepcopy(prior["response"])
                    if prior["status"] == "running":
                        raise GameError("REQUEST_RUNNING", 202)
                    raise GameError(prior.get("error", "REQUEST_FAILED"), prior.get("http_status", 502))
                if self.normal_busy():
                    raise GameError("GAME_BUSY")
                current_epoch, current_revision = self.conversation_state()
                if current_epoch and current_epoch != conversation_epoch:
                    raise GameError("STALE_CONVERSATION")
                if current_epoch and current_revision != normal_revision:
                    raise GameError("STALE_CONVERSATION")
                if self._closed_pages.get(page_id, 0) > self.clock():
                    raise GameError("GAME_CLOSED", 410)
                if self._session:
                    session = self._session
                    self._guard_owner(session, page_id, conversation_epoch)
                    prior = session.request_records.get(request_id)
                    if prior:
                        if prior["payload_hash"] != payload_hash:
                            raise GameError("REQUEST_ID_REUSED")
                        if prior["status"] == "committed":
                            return self._response_from_record(session, prior)
                        if prior["status"] == "running":
                            raise GameError("REQUEST_RUNNING", 202)
                        raise GameError(prior.get("error", "REQUEST_FAILED"), prior.get("http_status", 502))
                    raise GameError("GAME_BUSY")
                captured_seed = seed() if callable(seed) else seed
                session = GameSession(
                    game_id=str(uuid.uuid4()),
                    owner_page_id=page_id,
                    conversation_epoch=conversation_epoch,
                    normal_revision=normal_revision,
                    character_id=str(captured_seed.get("character_id", "")),
                    character_name=str(captured_seed.get("character_name", "")),
                    seed=copy.deepcopy(captured_seed),
                    started_at=_now_iso(),
                    lease_expires_at=self.clock() + LEASE_SECONDS,
                )
                session.request_records[request_id] = {"payload_hash": payload_hash, "status": "running", "started_at": _now_iso()}
                session.operation_count = 1
                self._session = session
                start_record = {
                    "payload_hash": payload_hash, "status": "running",
                    "expires_at": self.clock() + TOMBSTONE_SECONDS, "game_id": session.game_id,
                }
                self._start_records[key] = start_record
        try:
            llm = self._snapshot_llm(on_event)
            from .validation import validate_opening
            opening = normalize_opening(llm.opening(copy.deepcopy(session.seed)), session.character_name)
            validate_opening(opening)
            self._validate_limits(session)
            scene = {
                "location": opening["location"],
                "scene_time": opening.get("scene_time", ""),
                "entities": [{
                    "id": _id("entity"),
                    "key": entity["key"],
                    "name": entity["name"],
                    "kind": entity["kind"],
                    "perceived_by": entity["perceived_by"],
                    "description_by_view": entity["description_by_view"],
                } for entity in opening.get("entities", [])],
                "facts": {"player": [], "character": []},
                "hooks": [],
            }
            # The selected character is addressable for talk commands but is not a look target.
            scene["entities"].append({
                "id": "character",
                "key": "character",
                "name": session.character_name,
                "kind": "person",
                "is_character": True,
                "perceived_by": ["player", "character"],
                "description_by_view": {"player": "", "character": ""},
            })
            temp = copy.deepcopy(session)
            temp.scene = scene
            temp.memory = new_memory(scene)
            temp.world_turn = 1
            opening_event = self._world_event(
                temp,
                "environment",
                {"player": opening["narration"], "character": opening.get("character_event", "")},
                source="environment",
                actor_id="environment",
                important=True,
                world_turn=1,
            )
            events = [opening_event]
            for fact in opening.get("important_facts", []):
                for view in fact["known_by"]:
                    temp.memory["important_facts"][view].append({"text": fact["text"], "source_event_ids": [opening_event["id"]]})
            add_turn(temp.memory, 1, events)
            self._add_public(temp, "narration", opening["narration"])
            if on_event:
                on_event({"type": "narration_preview", "text": opening["narration"]})
            character_text = ""
            if opening["character_should_reply"] and opening.get("character_event"):
                character_text = self._character_reply(temp, llm)
                character_event = self._world_event(temp, "speech", {"player": character_text, "character": character_text}, source="character", actor_id="character", world_turn=1)
                events.append(character_event)
                temp.memory["recent_turns"][-1]["by_view"]["player"].append({"event_id": character_event["id"], "type": "speech", "text": character_text, "speaker": "character"})
                temp.memory["recent_turns"][-1]["by_view"]["character"].append({"event_id": character_event["id"], "type": "speech", "text": character_text, "speaker": "character"})
                self._add_public(temp, "character", character_text)
            choices = self._choices(temp, llm, character_text)
            self._install_choices(temp, choices, character_text)
            self._add_public(temp, "interaction", choices=choices)
            temp.status = "active"
            temp.revision = 1
            self._check_size(temp)
            result = self._view(temp, temp.public_entries, include_transcript=True)
            result["request_id"] = request_id
            with self._lock:
                if self._session is not session or session.status != "starting":
                    raise GameError("GAME_CLOSED", 410)
                self._finish_record(temp, request_id, result)
                self._check_size(temp)
                self._session = temp
                self._persist(temp)
                result["log_warning"] = temp.log_warning
                self._finish_record(temp, request_id, result)
                self._start_records[(page_id, request_id)].update({
                    "status": "committed", "expires_at": float("inf"), "response": copy.deepcopy(result),
                })
                return result
        except GameError as exc:
            with self._lock:
                if self._session is session:
                    self._fail_record(session, request_id, exc)
                    self._start_records[(page_id, request_id)].update({"status": "failed", "error": exc.code, "http_status": exc.status_code})
                    self._session = None
                else:
                    self._start_records[(page_id, request_id)].update({"status": "failed", "error": exc.code, "http_status": exc.status_code})
            raise
        except Exception as exc:
            error = GameError("INVALID_MODEL_OUTPUT", 502)
            with self._lock:
                if self._session is session:
                    self._fail_record(session, request_id, error)
                    self._start_records[(page_id, request_id)].update({"status": "failed", "error": error.code, "http_status": error.status_code})
                    self._session = None
                else:
                    self._start_records[(page_id, request_id)].update({"status": "failed", "error": error.code, "http_status": error.status_code})
            raise error from exc

    def view(self, game_id: str, page_id: str) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id)
            return self._view(session)

    def request_result(self, game_id: str, page_id: str, request_id: str) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id)
            record = session.request_records.get(request_id)
            if not record:
                raise GameError("REQUEST_NOT_FOUND", 404)
            result: dict[str, Any] = {"status": record["status"]}
            if record["status"] == "committed":
                result["result"] = self._response_from_record(session, record)
            elif record["status"] == "failed":
                result["error"] = record.get("error", "REQUEST_FAILED")
            return result

    def _target_interaction(self, session: GameSession, command: str, prompt: str, target_ids: list[str] | None = None) -> Interaction:
        entities = [
            entity for entity in session.scene.get("entities", [])
            if "player" in entity.get("perceived_by", [])
            and (target_ids is None or entity["id"] in target_ids)
            and (command != "look" or not entity.get("is_character"))
        ]
        choices = [Choice(
            _id("choice"), ordinal, entity["name"],
            {"kind": "select_target", "command": command, "target_id": entity["id"]},
        ) for ordinal, entity in enumerate(entities[:8], 1)]
        if not choices:
            prompt += " 対象を特定できない場合は、対象を文章で説明してください。"
        elif len(entities) > 8:
            prompt += " 一覧にない対象は名前を入力してください。"
        return self._make_interaction("target", prompt, choices, command)

    def _talk_content_interaction(self, session: GameSession, llm: Any, target_id: str) -> Interaction:
        entity = next((e for e in session.scene.get("entities", []) if e["id"] == target_id), None)
        if not entity or "player" not in entity.get("perceived_by", []):
            raise GameError("INVALID_CHOICE")
        choices = self._choices(session, llm, "", kind="content")
        for choice in choices:
            for step in choice.selection["steps"]:
                step["target_id"] = target_id
        return self._make_interaction("content", f"{entity['name']}に何と話しますか？", choices, "talk")

    def _chosen_input(self, session: GameSession, request_input: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        kind = request_input.get("kind")
        interaction_id = request_input.get("interaction_id")
        if session.pending is None or interaction_id != session.pending.id:
            raise GameError("STALE_REVISION")
        if kind == "choice":
            choice_id = request_input.get("choice_id")
            choice = next((c for c in session.pending.choices if c.id == choice_id), None)
            if not choice:
                raise GameError("INVALID_CHOICE")
            return "", copy.deepcopy(choice.selection), choice.label
        if kind != "text":
            raise GameError("INVALID_REQUEST", 422)
        text = request_input.get("text")
        if not isinstance(text, str) or len(text) > MAX_INPUT_CHARS:
            raise GameError("INVALID_REQUEST", 422)
        return text.strip(), None, None

    def submit(
        self,
        game_id: str,
        *,
        page_id: str,
        conversation_epoch: str,
        revision: int,
        request_id: str,
        request_input: dict[str, Any],
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        payload = {"page_id": page_id, "epoch": conversation_epoch, "revision": revision, "input": request_input}
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id, conversation_epoch)
            # Exact retries of a request already in flight must report
            # REQUEST_RUNNING even while its first copy owns the turn slot.
            if request_id in session.request_records:
                status, previous = self._record(session, request_id, payload)
                if status == "committed":
                    return previous
            if session.status != "active":
                raise GameError("GAME_BUSY")
            self._validate_limits(session)
            status, previous = self._record(session, request_id, payload)
            if status == "committed":
                return previous
            if revision != session.revision:
                session.request_records.pop(request_id, None)
                session.operation_count = max(0, session.operation_count - 1)
                raise GameError("STALE_REVISION")
            old_revision = session.revision
            session.status = "resolving"
            session.generation += 1
            generation = session.generation
            temp = copy.deepcopy(session)
            temp.status = "active"
        llm = self._snapshot_llm(on_event)
        try:
            text, selection, chosen_label = self._chosen_input(temp, request_input)
            if not text and selection is None:
                # Blank input re-displays the active card and does not create a world turn.
                with self._lock:
                    if self._session is not session or session.generation != generation or session.status != "resolving":
                        raise GameError("GAME_CLOSED", 410)
                    session.status = "active"
                    response = self._view(session, [], include_transcript=False)
                    response["request_id"] = request_id
                    self._finish_record(session, request_id, response)
                    return response
            normalized = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
            if selection is None and re.fullmatch(r"\d+", normalized):
                index = int(normalized)
                if index < 1 or not session.pending or index > len(session.pending.choices):
                    with self._lock:
                        if self._session is not session or session.generation != generation or session.status != "resolving":
                            raise GameError("GAME_CLOSED", 410)
                        session.status = "active"
                        response = self._view(session, [], include_transcript=False)
                        response["request_id"] = request_id
                        response["notice"] = "有効な候補番号を選んでください。"
                        self._finish_record(session, request_id, response)
                        return response
                choice = temp.pending.choices[index - 1]
                selection = copy.deepcopy(choice.selection)
                chosen_label = choice.label
            if selection and selection.get("kind") == "select_target":
                command = selection["command"]
                if command == "talk":
                    temp.pending = self._talk_content_interaction(temp, llm, selection["target_id"])
                    temp.revision += 1
                    self._add_public(temp, "interaction", text=temp.pending.prompt, choices=temp.pending.choices)
                    self._check_size(temp)
                    response = self._view(temp, temp.public_entries[-1:], include_transcript=False)
                    response["request_id"] = request_id
                    return self._commit_result(session, temp, generation, request_id, response)
                selection = {"kind": "perform", "steps": [{"kind": "action", "text": "対象を見る", "target_id": selection["target_id"]}]}
                text = "対象を見る"
            if text == "キャンセル" and selection is None:
                if not temp.base_choices:
                    raise GameError("INVALID_CHOICE")
                temp.pending = self._make_interaction(
                    "reply" if any(s["kind"] == "speech" for c in temp.base_choices for s in c.selection["steps"]) else "action",
                    None,
                    copy.deepcopy(temp.base_choices),
                )
                temp.revision += 1
                self._add_public(temp, "interaction", choices=temp.pending.choices)
                self._check_size(temp)
                response = self._view(temp, temp.public_entries[-1:], include_transcript=False)
                response["request_id"] = request_id
                return self._commit_result(session, temp, generation, request_id, response)
            if text == "見る" and selection is None:
                temp.pending = self._target_interaction(temp, "look", "何を見ますか？")
                temp.revision += 1
                self._add_public(temp, "interaction", text=temp.pending.prompt, choices=temp.pending.choices)
                self._check_size(temp)
                response = self._view(temp, temp.public_entries[-1:], include_transcript=False)
                response["request_id"] = request_id
                return self._commit_result(session, temp, generation, request_id, response)
            if text == "話す" and selection is None:
                temp.pending = self._target_interaction(temp, "talk", "誰に話しかけますか？", ["character"])
                temp.revision += 1
                self._add_public(temp, "interaction", text=temp.pending.prompt, choices=temp.pending.choices)
                self._check_size(temp)
                response = self._view(temp, temp.public_entries[-1:], include_transcript=False)
                response["request_id"] = request_id
                return self._commit_result(session, temp, generation, request_id, response)

            if not text and selection is not None:
                for step in selection.get("steps", []):
                    if step.get("target_id") and not self._visible_entity(temp, step["target_id"], "player"):
                        raise GameError("INVALID_CHOICE")
            may_hook = (
                temp.world_turn - int(temp.scene.get("last_hook_world_turn", -10_000)) >= HOOK_INTERVAL_TURNS
                and not any(h.get("status") == "open" for h in temp.scene.get("hooks", []))
                and not (temp.pending and temp.pending.kind in {"target", "content"})
            )
            payload_for_llm = {
                "scene": copy.deepcopy(temp.scene),
                "gm_context": view_context(temp.memory, "player", include_entity_keys=True),
                "character_context": view_context(temp.memory, "character"),
                "input": text,
                "selection": selection,
                "may_introduce_hook": may_hook,
                "known_target_keys": [e["key"] for e in temp.scene.get("entities", []) if "player" in e.get("perceived_by", [])],
            }
            output = normalize_resolution(llm.resolve(payload_for_llm), temp.scene)
            from .validation import validate_resolution
            validate_resolution(output)
            if output["kind"] == "clarify":
                allowed = {
                    entity["key"]: entity["id"]
                    for entity in temp.scene.get("entities", [])
                    if "player" in entity.get("perceived_by", [])
                }
                target_ids = output.get("target_keys", [])
                if any(target_key not in allowed for target_key in target_ids):
                    raise GameError("INVALID_MODEL_OUTPUT", 502)
                command = output["command"]
                resolved_target_ids = [allowed[target_key] for target_key in target_ids]
                temp.pending = self._target_interaction(temp, command, output["question"], resolved_target_ids)
                temp.revision += 1
                self._add_public(temp, "interaction", text=temp.pending.prompt, choices=temp.pending.choices)
                self._check_size(temp)
                response = self._view(temp, temp.public_entries[-1:], include_transcript=False)
                response["request_id"] = request_id
                return self._commit_result(session, temp, generation, request_id, response)

            # New entities become addressable only after this turn is accepted,
            # and only in viewpoints that received their introduction.
            additions = output.get("entity_additions", [])
            existing_keys = {entity["key"] for entity in temp.scene.get("entities", [])}
            for addition in additions:
                key = addition["key"]
                name = addition["name"]
                if key in existing_keys:
                    raise GameError("INVALID_MODEL_OUTPUT", 502)
                if "player" in addition["perceived_by"] and name not in (output.get("narration", "") + (output.get("hook") or "")):
                    raise GameError("INVALID_MODEL_OUTPUT", 502)
                if "character" in addition["perceived_by"] and name not in output.get("character_event", ""):
                    raise GameError("INVALID_MODEL_OUTPUT", 502)
                existing_keys.add(key)
                temp.scene.setdefault("entities", []).append({
                    "id": _id("entity"),
                    "key": key,
                    "name": name,
                    "kind": addition["kind"],
                    "perceived_by": list(addition["perceived_by"]),
                    "description_by_view": copy.deepcopy(addition["description_by_view"]),
                })
            temp.memory["current_scene"] = copy.deepcopy(temp.scene)

            output_intent = output["intent"]
            steps = selection.get("steps", []) if selection else []
            if not selection:
                if output_intent == "speech":
                    # A direct line reaches the character verbatim; classification cannot rewrite it.
                    steps = [{"kind": "speech", "text": text, "target_id": output.get("target_key") or "character"}]
                elif output_intent == "mixed":
                    speech = output.get("speech_text", "")
                    action = output.get("action_text", "")
                    if not speech or speech not in text or not action or action not in text:
                        raise GameError("INVALID_MODEL_OUTPUT", 502)
                    steps = [
                        {"kind": "action", "text": action, "target_id": output.get("target_key")},
                        {"kind": "speech", "text": speech, "target_id": "character"},
                    ]
                elif output_intent == "observation":
                    steps = [{"kind": "action", "text": text, "target_id": output.get("target_key")}]
                else:
                    action = output.get("action_text") or text
                    if action not in text and action != text:
                        raise GameError("INVALID_MODEL_OUTPUT", 502)
                    steps = [{"kind": "action", "text": text, "target_id": output.get("target_key")}]
            if not steps:
                raise GameError("INVALID_MODEL_OUTPUT", 502)
            for step in steps:
                if step.get("kind") not in {"speech", "action"} or not _text(step.get("text")):
                    raise GameError("INVALID_MODEL_OUTPUT", 502)
                target_id = step.get("target_id")
                if target_id and not self._visible_entity(temp, target_id, "player"):
                    raise GameError("INVALID_MODEL_OUTPUT", 502)
            if output.get("hook") and not may_hook:
                raise GameError("INVALID_MODEL_OUTPUT", 502)

            self._validate_limits(temp)
            next_turn = temp.world_turn + 1
            world_events: list[dict[str, Any]] = []
            input_display = chosen_label if chosen_label is not None else text
            self._add_public(temp, "player", input_display)
            spoken_steps = [step for step in steps if step["kind"] == "speech"]
            action_steps = [step for step in steps if step["kind"] == "action"]
            if spoken_steps:
                exact_speech = " ".join(step["text"] for step in spoken_steps)
                world_events.append(self._world_event(
                    temp, "speech", {"player": exact_speech, "character": exact_speech},
                    source="player", actor_id="player",
                    target_ids=[s["target_id"] for s in spoken_steps if s.get("target_id")],
                    world_turn=next_turn,
                ))
            if action_steps:
                original_action = "／".join(step["text"] for step in action_steps)
                character_event = output.get("character_event", "")
                world_events.append(self._world_event(
                    temp, "action", {"player": original_action, "character": character_event},
                    source="player", actor_id="player",
                    target_ids=[s["target_id"] for s in action_steps if s.get("target_id")],
                    outcome="occurred" if output.get("action_succeeded") else "failed_attempt",
                    important=True,
                    world_turn=next_turn,
                ))
            if output.get("narration"):
                self._add_public(temp, "narration", output["narration"])
                world_events.append(self._world_event(
                    temp, "environment", {"player": output["narration"], "character": output.get("character_event", "") if not action_steps else ""},
                    source="environment", actor_id="environment", world_turn=next_turn,
                ))
            if output.get("hook"):
                hook_text = output["hook"]
                self._add_public(temp, "narration", hook_text)
                hook_event = self._world_event(
                    temp, "environment", {"player": hook_text, "character": output.get("character_event", "")},
                    source="environment", actor_id="environment", important=True, world_turn=next_turn,
                )
                world_events.append(hook_event)
                hook_id = _id("hook")
                temp.scene.setdefault("hooks", []).append({"id": hook_id, "event_id": hook_event["id"], "introduced_turn": next_turn, "status": "open"})
                temp.scene["last_hook_world_turn"] = next_turn
            temp.world_turn = next_turn
            add_turn(temp.memory, next_turn, world_events)
            for fact in output.get("important_facts", []):
                fact_id = world_events[-1]["id"] if world_events else _id("fact-event")
                for view in fact["known_by"]:
                    temp.memory["important_facts"][view].append({"text": fact["text"], "source_event_ids": [fact_id]})
            for hook in temp.scene.get("hooks", []):
                if hook.get("status") == "open" and next_turn - hook["introduced_turn"] >= 2:
                    hook["status"] = "ignored"

            narration = "\n\n".join(output[key] for key in ("narration", "hook") if output.get(key))
            if on_event and narration:
                on_event({"type": "narration_preview", "text": narration})
            character_text = ""
            can_character_perceive = any(event["text_by_view"].get("character") for event in world_events)
            if output["character_should_reply"] and can_character_perceive:
                character_text = self._character_reply(temp, llm)
                char_event = self._world_event(temp, "speech", {"player": character_text, "character": character_text}, source="character", actor_id="character", world_turn=next_turn)
                world_events.append(char_event)
                if temp.memory["recent_turns"] and temp.memory["recent_turns"][-1]["world_turn"] == next_turn:
                    for view in ("player", "character"):
                        temp.memory["recent_turns"][-1]["by_view"][view].append({"event_id": char_event["id"], "type": "speech", "text": character_text, "speaker": "character"})
                else:
                    add_turn(temp.memory, next_turn, [char_event])
                self._add_public(temp, "character", character_text)

            compact_if_needed(temp.memory, next_turn, lambda view, payload: llm.summarize({"view": view, **payload}))
            choices = self._choices(temp, llm, character_text)
            self._install_choices(temp, choices, character_text)
            self._add_public(temp, "interaction", choices=choices)
            temp.revision += 1
            self._check_size(temp)
            new_entries = temp.public_entries[len(session.public_entries):]
            response = self._view(temp, new_entries, include_transcript=False)
            response["request_id"] = request_id
            return self._commit_result(session, temp, generation, request_id, response)
        except GameError as exc:
            with self._lock:
                if self._session is session and session.generation == generation and session.status == "resolving":
                    session.status = "active"
                    self._fail_record(session, request_id, exc)
            raise
        except Exception as exc:
            error = GameError("INVALID_MODEL_OUTPUT", 502)
            with self._lock:
                if self._session is session and session.generation == generation and session.status == "resolving":
                    session.status = "active"
                    self._fail_record(session, request_id, error)
            raise error from exc

    def _commit_result(self, original: GameSession, updated: GameSession, generation: int, request_id: str, response: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._session is not original or original.generation != generation or original.status != "resolving":
                raise GameError("GAME_CLOSED", 410)
            updated.status = "active"
            updated.generation = generation
            updated.lease_expires_at = self.clock() + LEASE_SECONDS
            updated.request_records[request_id] = copy.deepcopy(original.request_records[request_id])
            self._finish_record(updated, request_id, response)
            self._check_size(updated)
            self._session = updated
            self._persist(updated)
            response["log_warning"] = updated.log_warning
            self._finish_record(updated, request_id, response)
            return response

    @staticmethod
    def _visible_entity(session: GameSession, entity_id: str, view: str) -> dict[str, Any] | None:
        return next((e for e in session.scene.get("entities", []) if e["id"] == entity_id and view in e.get("perceived_by", [])), None)

    def suspend(self, game_id: str, *, page_id: str, revision: int, request_id: str) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id)
            payload = {"page_id": page_id, "revision": revision, "mode": "suspend"}
            status, response = self._record(session, request_id, payload)
            if status == "committed":
                return response
            if revision != session.revision:
                session.request_records.pop(request_id, None)
                session.operation_count -= 1
                raise GameError("STALE_REVISION")
            if session.status not in {"active", "resolving"}:
                self._fail_record(session, request_id, GameError("GAME_BUSY"))
                raise GameError("GAME_BUSY")
            session.generation += 1
            session.status = "suspended"
            self._fail_running_records(session, "GAME_SUSPENDED", 409)
            session.revision += 1
            session.lease_expires_at = self.clock() + LEASE_SECONDS
            result = self._view(session, [], include_transcript=False)
            result["request_id"] = request_id
            self._finish_record(session, request_id, result)
            return result

    def resume(self, game_id: str, *, page_id: str, conversation_epoch: str, revision: int, request_id: str) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id, conversation_epoch)
            payload = {"page_id": page_id, "epoch": conversation_epoch, "revision": revision, "mode": "resume"}
            status, response = self._record(session, request_id, payload)
            if status == "committed":
                return response
            if revision != session.revision:
                session.request_records.pop(request_id, None)
                session.operation_count -= 1
                raise GameError("STALE_REVISION")
            if session.status != "suspended":
                self._fail_record(session, request_id, GameError("GAME_BUSY"))
                raise GameError("GAME_BUSY")
            session.status = "active"
            session.revision += 1
            session.generation += 1
            session.lease_expires_at = self.clock() + LEASE_SECONDS
            if session.pending:
                session.pending.id = _id("interaction")
            result = self._view(session, [], include_transcript=False)
            result["request_id"] = request_id
            self._finish_record(session, request_id, result)
            return result

    def lease(self, game_id: str, page_id: str) -> None:
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id)
            session.lease_expires_at = self.clock() + LEASE_SECONDS

    def close(self, game_id: str, *, page_id: str, request_id: str = "") -> None:
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session or session.game_id != game_id:
                closed = self._closed_games.get(game_id)
                if closed and closed[0] == page_id:
                    return
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id)
            self._close_locked(session, "client_close")

    def close_page(self, page_id: str) -> None:
        with self._lock:
            self._prune_locked()
            self._closed_pages[page_id] = self.clock() + TOMBSTONE_SECONDS
            session = self._session
            if session and session.owner_page_id == page_id:
                self._close_locked(session, "pagehide")

    def audio_payload(self, game_id: str, page_id: str, event_id: str, revision: int, request_id: str, retry: bool = False) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id)
            if session.status not in {"active", "suspended"}:
                raise GameError("GAME_BUSY")
            if not request_id or len(request_id) > 128:
                raise GameError("INVALID_REQUEST", 422)
            event = next((e for e in session.public_entries if e["id"] == event_id and e["type"] == "character"), None)
            if not event or not event.get("text"):
                raise GameError("INVALID_AUDIO_EVENT")
            if event.get("revision") != revision:
                raise GameError("STALE_REVISION")
            payload_hash = _stable_hash({"event_id": event_id, "revision": revision, "retry": bool(retry)})
            prior_request = next((item for item in session.audio_records.values() if item.get("request_id") == request_id), None)
            if prior_request:
                if prior_request.get("payload_hash") != payload_hash:
                    raise GameError("REQUEST_ID_REUSED")
                existing = copy.deepcopy(prior_request)
                existing["already_started"] = True
                return existing
            key = f"{event_id}:{request_id}"
            if not retry and any(k.startswith(f"{event_id}:") for k in session.audio_records):
                prior_key = next(k for k in session.audio_records if k.startswith(f"{event_id}:"))
                existing = copy.deepcopy(session.audio_records[prior_key])
                existing["already_started"] = True
                return existing
            voice = session.seed.get("voice", {})
            record = {
                "text": event["text"],
                "voice_path": voice.get("voice_path"),
                "embedding_path": voice.get("embedding_path"),
                "tts_available": bool(voice.get("tts_available")),
                "request_id": request_id,
                "retry": retry,
                "payload_hash": payload_hash,
                "event_id": event_id,
                "revision": revision,
                "already_started": False,
                "audio_req_id": _id("gameaudio"),
            }
            if not record["tts_available"]:
                raise GameError("TTS_UNAVAILABLE", 400)
            session.audio_records[key] = copy.deepcopy(record)
            try:
                self._check_size(session)
            except GameError:
                session.audio_records.pop(key, None)
                raise
            return record

    def export_log(self, game_id: str, page_id: str) -> tuple[bytes, str]:
        with self._lock:
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id)
            from .transcript import GameTranscriptWriter
            return GameTranscriptWriter().encode(session), f"{session.game_id}.game.json"

    def save_log(self, game_id: str, page_id: str) -> str | None:
        with self._lock:
            session = self._session
            if not session or session.game_id != game_id:
                raise GameError("GAME_CLOSED", 410)
            self._guard_owner(session, page_id)
            self._persist(session)
            return getattr(session, "log_warning", None)

    def save_current_log(self) -> str | None:
        with self._lock:
            self._prune_locked()
            if not self._session:
                return None
            self._persist(self._session)
            return getattr(self._session, "log_warning", None)

    def current(self, page_id: str) -> GameSession | None:
        with self._lock:
            self._prune_locked()
            if self._session and self._session.owner_page_id == page_id:
                return copy.deepcopy(self._session)
            return None

    def activity(self) -> str | None:
        with self._lock:
            self._prune_locked()
            return self._session.status if self._session else None

    def transition(self, page_id: str | None, epoch: str | None) -> None:
        """End the retained session before a validated select/prompt/log transition."""
        with self._lock:
            self._prune_locked()
            session = self._session
            if not session:
                return
            if not page_id or session.owner_page_id != page_id:
                raise GameError("OWNER_MISMATCH")
            if epoch and session.conversation_epoch != epoch:
                raise GameError("STALE_CONVERSATION")
            if session.status in {"starting", "resolving"}:
                raise GameError("GAME_BUSY")
            self._close_locked(session, "conversation_transition")
