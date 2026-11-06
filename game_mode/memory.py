"""Ephemeral game memory with separate player and character projections."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Callable

from .models import GameError
from .validation import validate_summary

RECENT_TURNS = 6
COMPACT_TURNS = 12
COMPACT_CHARS = 24_000
HARD_CONTEXT_CHARS = 48_000
SUMMARY_MAX_CHARS = 2_000


def new_memory(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_scene": deepcopy(scene),
        "important_facts": {"player": [], "character": []},
        "summaries": {"player": "", "character": ""},
        "recent_turns": [],
        "event_ledger": [],
        "summarized_through_turn": 0,
    }


def add_turn(memory: dict[str, Any], world_turn: int, events: list[dict[str, Any]]) -> None:
    by_view = {"player": [], "character": []}
    for event in events:
        memory["event_ledger"].append(deepcopy(event))
        text_by_view = event.get("text_by_view", {})
        for view in by_view:
            text = text_by_view.get(view)
            if text:
                projected = {"event_id": event["id"], "type": event["kind"], "text": text}
                if event["kind"] == "speech" and event.get("actor_id") in {"player", "character"}:
                    projected["speaker"] = event["actor_id"]
                by_view[view].append(projected)
                if event.get("important"):
                    memory["important_facts"][view].append({
                        "text": text[:500],
                        "source_event_ids": [event["id"]],
                    })
    if any(by_view.values()):
        memory["recent_turns"].append({"world_turn": world_turn, "by_view": by_view})
    memory["current_scene"] = deepcopy(memory.get("current_scene", {}))


def view_context(memory: dict[str, Any], view: str, *, include_entity_keys: bool = False) -> dict[str, Any]:
    """Project only information perceived by the requested viewpoint."""
    raw_scene = memory.get("current_scene", {})
    entities = raw_scene.get("entities", [])
    projected_entities = []
    for entity in entities:
        if view not in entity.get("perceived_by", []):
            continue
        description = entity.get("description_by_view", {}).get(view, "")
        projected = {"name": entity.get("name", ""), "kind": entity.get("kind", ""), "description": description}
        if include_entity_keys:
            projected["key"] = entity.get("key", "")
        projected_entities.append(projected)
    # Do not pass hook bookkeeping, IDs, other-view descriptions, or raw
    # perception flags to role prompts. Those belong to the engine/GM only.
    scene = {
        "location": raw_scene.get("location", ""),
        "scene_time": raw_scene.get("scene_time", ""),
        "entities": projected_entities,
        "facts": [{"text": item.get("text", "")} for item in memory.get("important_facts", {}).get(view, [])],
    }
    recent = []
    for turn in memory.get("recent_turns", [])[-RECENT_TURNS:]:
        events = turn.get("by_view", {}).get(view, [])
        if events:
            recent.append({
                "world_turn": turn["world_turn"],
                "events": [
                    {"type": event.get("type", ""), "text": event.get("text", ""),
                     **({"speaker": event["speaker"]} if event.get("speaker") in {"player", "character"} else {})}
                    for event in events
                ],
            })
    return {
        "scene": scene,
        "important_facts": [{"text": item.get("text", "")} for item in memory.get("important_facts", {}).get(view, [])],
        "summary": memory.get("summaries", {}).get(view, ""),
        "recent_turns": recent,
    }


def compact_if_needed(
    memory: dict[str, Any],
    world_turn: int,
    summarize: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    """Summarize old turns independently per viewpoint on the speculative copy."""
    turns = memory.get("recent_turns", [])
    # The ledger is an archive, never a role prompt. Counting it makes every
    # sufficiently long game hit the limit even after successful compaction.
    # These limits are characters, not UTF-8 bytes (notably for Japanese).
    context_size = len(json.dumps({
        key: memory.get(key) for key in
        ("current_scene", "important_facts", "summaries", "recent_turns")
    }, ensure_ascii=False, separators=(",", ":")))
    if len(turns) <= COMPACT_TURNS and context_size <= COMPACT_CHARS:
        return
    old = turns[:-RECENT_TURNS]
    if not old:
        if context_size > HARD_CONTEXT_CHARS:
            raise GameError("MEMORY_COMPACTION_FAILED", 413)
        return
    outputs: dict[str, str] = {}
    for view in ("player", "character"):
        visible = [
            {"world_turn": turn["world_turn"], "events": deepcopy(turn["by_view"].get(view, []))}
            for turn in old if turn["by_view"].get(view)
        ]
        if not visible:
            outputs[view] = memory["summaries"].get(view, "")
            continue
        try:
            result = summarize(view, {
                "previous_summary": memory["summaries"].get(view, ""),
                "turns": visible,
                "important_facts": deepcopy(memory["important_facts"].get(view, [])),
            })
            validate_summary(result)
            text = result["text"]
            source_ids = set(result["source_event_ids"])
            allowed_ids = {
                event["event_id"] for turn in visible for event in turn["events"]
            }
            if len(text) > SUMMARY_MAX_CHARS or not source_ids <= allowed_ids:
                raise ValueError("invalid summary sources or length")
        except (GameError, ValueError) as exc:
            # Keep both projections intact and retry on the next turn when
            # there is still room. The LLM adapter raises GameError after its
            # own retries, rather than returning an invalid summary object.
            if context_size > HARD_CONTEXT_CHARS:
                raise GameError("MEMORY_COMPACTION_FAILED", 413) from exc
            return
        outputs[view] = text
    memory["summaries"] = outputs
    memory["recent_turns"] = turns[-RECENT_TURNS:]
    memory["summarized_through_turn"] = old[-1]["world_turn"]
