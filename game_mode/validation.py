"""Structural validation for untrusted model output."""

from __future__ import annotations

from typing import Any


def _object(value: Any, allowed: set[str], required: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - allowed or not required <= set(value):
        raise ValueError(f"invalid {name} fields")
    return value


def _text(value: Any, name: str, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not allow_empty and not value.strip()):
        raise ValueError(f"invalid {name}")
    return value.strip()


def validate_opening(obj: dict[str, Any]) -> None:
    _object(
        obj,
        {"location", "scene_time", "narration", "character_should_reply", "character_event", "entities", "important_facts"},
        {"location", "narration", "character_should_reply"},
        "opening",
    )
    _text(obj.get("location"), "location", 120)
    _text(obj.get("scene_time", ""), "scene_time", 80, allow_empty=True)
    _text(obj.get("narration"), "narration", 300)
    if not isinstance(obj.get("character_should_reply"), bool):
        raise ValueError("invalid reply flag")
    _text(obj.get("character_event", ""), "character_event", 600, allow_empty=True)
    entities = obj.get("entities", [])
    facts = obj.get("important_facts", [])
    if not isinstance(entities, list) or len(entities) > 8 or not isinstance(facts, list) or len(facts) > 20:
        raise ValueError("invalid opening lists")
    keys: set[str] = set()
    for entity in entities:
        _object(entity, {"key", "name", "kind", "perceived_by", "description_by_view"},
                {"key", "name", "kind", "perceived_by", "description_by_view"}, "entity")
        key = _text(entity.get("key"), "entity key", 32)
        name = _text(entity.get("name"), "entity name", 80)
        if key in keys or name not in obj["narration"]:
            raise ValueError("entity must be uniquely introduced in narration")
        keys.add(key)
        if entity.get("kind") not in {"object", "person", "place"}:
            raise ValueError("invalid entity kind")
        perceived = entity.get("perceived_by")
        if not isinstance(perceived, list) or not set(perceived) <= {"player", "character"}:
            raise ValueError("invalid entity perception")
        descriptions = entity.get("description_by_view")
        if not isinstance(descriptions, dict):
            raise ValueError("invalid entity descriptions")
        for view in perceived:
            _text(descriptions.get(view), f"{view} description", 300)
    for fact in facts:
        _object(fact, {"text", "known_by"}, {"text", "known_by"}, "fact")
        _text(fact.get("text"), "fact", 500)
        known = fact.get("known_by")
        if not isinstance(known, list) or not known or not set(known) <= {"player", "character"}:
            raise ValueError("invalid fact visibility")


def validate_resolution(obj: dict[str, Any]) -> None:
    kind = obj.get("kind")
    if kind == "clarify":
        _object(obj, {"kind", "question", "command", "target_keys"}, {"kind", "question", "command"}, "clarification")
        _text(obj.get("question"), "question", 300)
        if obj.get("command") not in {"look", "talk", "content", "action"}:
            raise ValueError("invalid clarification command")
        targets = obj.get("target_keys", [])
        if (not isinstance(targets, list) or len(targets) > 8
                or any(not isinstance(k, str) or not k or len(k) > 32 for k in targets)
                or len(targets) != len(set(targets))):
            raise ValueError("invalid clarification targets")
        return
    if kind != "resolve" or obj.get("intent") not in {"speech", "action", "mixed", "observation"}:
        raise ValueError("invalid resolution kind")
    _object(
        obj,
        {"kind", "intent", "speech_text", "action_text", "target_key", "narration", "action_succeeded",
         "character_should_reply", "character_event", "hook", "important_facts", "entity_additions"},
        {"kind", "intent", "action_succeeded", "character_should_reply"},
        "resolution",
    )
    _text(obj.get("speech_text", ""), "speech", 2000, allow_empty=True)
    _text(obj.get("action_text", ""), "action", 2000, allow_empty=True)
    target_key = obj.get("target_key")
    if target_key is not None:
        _text(target_key, "target", 32)
    _text(obj.get("narration", ""), "narration", 1000, allow_empty=True)
    _text(obj.get("character_event", ""), "character event", 1000, allow_empty=True)
    if not isinstance(obj.get("action_succeeded"), bool) or not isinstance(obj.get("character_should_reply"), bool):
        raise ValueError("invalid outcome flags")
    hook = obj.get("hook")
    if hook is not None:
        _text(hook, "hook", 500)
    additions = obj.get("entity_additions", [])
    if not isinstance(additions, list) or len(additions) > 2:
        raise ValueError("invalid entity additions")
    keys: set[str] = set()
    for entity in additions:
        _object(
            entity,
            {"key", "name", "kind", "perceived_by", "description_by_view"},
            {"key", "name", "kind", "perceived_by", "description_by_view"},
            "entity addition",
        )
        key = _text(entity["key"], "entity key", 32)
        _text(entity["name"], "entity name", 80)
        if key in keys or entity["kind"] not in {"object", "person", "place"}:
            raise ValueError("invalid entity addition identity")
        keys.add(key)
        perceived = entity["perceived_by"]
        if (not isinstance(perceived, list) or not perceived or len(perceived) != len(set(perceived))
                or not set(perceived) <= {"player", "character"}):
            raise ValueError("invalid entity addition perception")
        descriptions = entity["description_by_view"]
        if not isinstance(descriptions, dict) or not set(perceived) <= set(descriptions):
            raise ValueError("invalid entity addition descriptions")
        for view in perceived:
            _text(descriptions.get(view), f"{view} entity description", 300)
    facts = obj.get("important_facts", [])
    if not isinstance(facts, list) or len(facts) > 10:
        raise ValueError("invalid facts")
    for fact in facts:
        _object(fact, {"text", "known_by"}, {"text", "known_by"}, "fact")
        _text(fact["text"], "fact", 500)
        known = fact["known_by"]
        if not isinstance(known, list) or not known or not set(known) <= {"player", "character"}:
            raise ValueError("invalid fact visibility")


def validate_choices(obj: dict[str, Any]) -> None:
    _object(obj, {"choices"}, {"choices"}, "choices")
    choices = obj.get("choices")
    if not isinstance(choices, list) or len(choices) != 3:
        raise ValueError("exactly three choices required")
    for choice in choices:
        _object(choice, {"steps"}, {"steps"}, "choice")
        steps = choice.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 2:
            raise ValueError("invalid choice steps")
        for step in steps:
            if step.get("kind") not in {"speech", "action"}:
                raise ValueError("invalid step")
            _object(step, {"kind", "text", "target_key"}, {"kind", "text"}, "step")
            _text(step.get("text"), "step text", 1000)
            target_key = step.get("target_key")
            if target_key is not None:
                _text(target_key, "step target", 32)


def validate_summary(obj: dict[str, Any]) -> None:
    _object(obj, {"text", "source_event_ids"}, {"text", "source_event_ids"}, "summary")
    _text(obj.get("text"), "summary", 2000, allow_empty=True)
    ids = obj.get("source_event_ids")
    if not isinstance(ids, list) or len(ids) > 500 or any(not isinstance(item, str) for item in ids):
        raise ValueError("invalid summary source ids")
