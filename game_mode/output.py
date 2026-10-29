"""Conservative preparation of optional LLM output before strict engine validation.

Only discard unsupported metadata or candidates. Never invent perception,
truncate executable text, or turn an unknown target into an untargeted action.
"""

from typing import Any

from .validation import validate_choices, validate_opening, validate_resolution


def normalize_opening(value: dict[str, Any], character_name: str) -> dict[str, Any]:
    result = {key: value[key] for key in (
        "location", "scene_time", "narration", "character_should_reply", "character_event",
    ) if key in value}
    result.setdefault("location", "現在の場所")
    result.setdefault("character_should_reply", False)
    result["entities"] = []
    result["important_facts"] = []
    # Essential scene content must still be usable. Do not guess a missing scene.
    validate_opening(result)
    entities = value.get("entities", [])
    for raw in entities if isinstance(entities, list) else []:
        if not isinstance(raw, dict) or raw.get("name") == character_name:
            continue
        entity = {key: raw[key] for key in ("name", "kind", "perceived_by", "description_by_view") if key in raw}
        # Retain usable keys for existing adapters, assigning missing/duplicate keys here.
        used_keys = {old["key"] for old in result["entities"]} | {"character"}
        key = raw.get("key")
        if not isinstance(key, str) or not key.strip() or len(key) > 32 or key in used_keys:
            ordinal = 1
            while f"e{ordinal}" in used_keys:
                ordinal += 1
            key = f"e{ordinal}"
        entity["key"] = key
        if any(old["name"] == entity.get("name") for old in result["entities"]):
            continue
        try:
            validate_opening({**result, "entities": [entity]})
        except (ValueError, TypeError):
            continue
        result["entities"].append(entity)
        if len(result["entities"]) == 8:
            break
    facts = value.get("important_facts", [])
    for raw in facts if isinstance(facts, list) else []:
        if not isinstance(raw, dict):
            continue
        fact = {key: raw[key] for key in ("text", "known_by") if key in raw}
        try:
            validate_opening({**result, "important_facts": [fact]})
        except (ValueError, TypeError):
            continue
        result["important_facts"].append(fact)
        if len(result["important_facts"]) == 20:
            break
    return result


def normalize_resolution(value: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    if value.get("kind") != "resolve":
        return value
    result = {**value, "entity_additions": []}
    # Optional entity metadata cannot invalidate an otherwise valid resolution.
    # Core action, visibility facts and outcome flags retain strict validation.
    validate_resolution(result)
    used_keys = {entity["key"] for entity in scene.get("entities", [])}
    additions = value.get("entity_additions", [])
    for raw in additions if isinstance(additions, list) else []:
        if not isinstance(raw, dict):
            continue
        entity = {key: raw[key] for key in (
            "key", "name", "kind", "perceived_by", "description_by_view",
        ) if key in raw}
        try:
            validate_resolution({**result, "entity_additions": [entity]})
            if entity["key"] in used_keys:
                continue
            perceived = entity["perceived_by"]
            if "player" in perceived and entity["name"] not in (result.get("narration", "") + (result.get("hook") or "")):
                continue
            if "character" in perceived and entity["name"] not in result.get("character_event", ""):
                continue
        except (ValueError, TypeError):
            continue
        used_keys.add(entity["key"])
        result["entity_additions"].append(entity)
        if len(result["entity_additions"]) == 2:
            break
    return result


def prepare_choices(value: Any, scene: dict[str, Any], *, content_only: bool = False) -> tuple[dict[str, Any], bool]:
    visible = {entity["key"] for entity in scene.get("entities", []) if "player" in entity.get("perceived_by", [])}
    raw_choices = value.get("choices", []) if isinstance(value, dict) else []
    accepted = []
    seen = set()

    def accept(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        # Accept the compact schema as well as the previous steps schema.
        steps = raw.get("steps", [raw])
        if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
            return
        step = {key: steps[0][key] for key in ("kind", "text", "target_key") if key in steps[0]}
        if isinstance(step.get("text"), str):
            step["text"] = step["text"].strip()
        candidate = {"steps": [step]}
        try:
            validate_choices({"choices": [candidate] * 3})
            if content_only and step["kind"] != "speech":
                return
            if step.get("target_key") and step["target_key"] not in visible:
                return
            identity = (step["kind"], step["text"], step.get("target_key"))
            if identity in seen:
                return
        except (ValueError, TypeError):
            return
        seen.add(identity)
        accepted.append(candidate)

    for raw in raw_choices if isinstance(raw_choices, list) else []:
        accept(raw)
        if len(accepted) == 3:
            break
    supplemented = len(accepted) < 3
    for text in ("もう少し詳しく聞かせて。", "別の方法を考えたい。", "少し考える時間がほしい。"):
        if len(accepted) == 3:
            break
        accept({"kind": "speech", "text": text})
    return {"choices": accepted}, supplemented
