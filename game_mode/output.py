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


# --- Lenient coercion of raw model output (runs before normalize/validate) ---

_BOOL_WORDS = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
_VIEW_NAMES = {"player", "character"}
_OPENING_FIELDS = ("location", "scene_time", "narration", "character_should_reply", "character_event",
                   "entities", "important_facts")
_RESOLUTION_FIELDS = ("intent", "speech_text", "action_text", "target_key", "narration", "action_succeeded",
                      "character_should_reply", "character_event", "hook", "important_facts", "entity_additions")
_SPEECH_WORDS = {"speech", "say", "talk", "speak", "dialogue", "line", "発話", "セリフ"}
_ACTION_WORDS = {"action", "act", "do", "move", "行動"}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return _BOOL_WORDS.get(value.strip().casefold(), default)
    return default


def _coerce_text(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return "".join(value)
    return value


def _coerce_optional_text(value: Any) -> Any:
    text = _coerce_text(value)
    if isinstance(text, str) and text.strip().casefold() in {"", "null", "none", "なし"}:
        return None
    return text


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def _coerce_views(value: Any) -> Any:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return value
    seen: list[str] = []
    for item in value:
        name = item.strip().casefold() if isinstance(item, str) else item
        if name == "both":
            for both in ("player", "character"):
                if both not in seen:
                    seen.append(both)
        elif name not in seen:
            seen.append(name)
    return seen


def _coerce_fact(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    text = _coerce_text(raw.get("text", raw.get("fact")))
    known = _coerce_views(raw.get("known_by", raw.get("perceived_by")))
    if not isinstance(text, str) or not text.strip() or not isinstance(known, list) or not known:
        return None
    if not set(known) <= _VIEW_NAMES:
        return None
    return {"text": text, "known_by": known}


def _coerce_entity(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    entity = {key: raw[key] for key in ("key", "name", "kind", "perceived_by", "description_by_view") if key in raw}
    if "perceived_by" in entity:
        entity["perceived_by"] = _coerce_views(entity["perceived_by"])
    if isinstance(entity.get("kind"), str):
        entity["kind"] = entity["kind"].strip().casefold()
    if isinstance(entity.get("name"), str):
        entity["name"] = entity["name"].strip()
    descriptions = entity.get("description_by_view")
    if isinstance(descriptions, dict):
        entity["description_by_view"] = {
            view: _coerce_text(text) for view, text in descriptions.items() if view in _VIEW_NAMES
        }
    return entity


def _coerce_opening(value: dict[str, Any]) -> dict[str, Any]:
    out = {key: value[key] for key in _OPENING_FIELDS if key in value}
    for key in ("location", "scene_time", "narration", "character_event"):
        if key in out:
            out[key] = _coerce_text(out[key])
    out["character_should_reply"] = _coerce_bool(value.get("character_should_reply"), bool(out.get("character_event")))
    out["entities"] = [entity for entity in (_coerce_entity(raw) for raw in _coerce_list(value.get("entities"))) if entity]
    out["important_facts"] = [fact for fact in (_coerce_fact(raw) for raw in _coerce_list(value.get("important_facts"))) if fact]
    return out


def _coerce_resolution(value: dict[str, Any]) -> dict[str, Any]:
    kind = value.get("kind")
    kind = kind.strip().casefold() if isinstance(kind, str) else kind
    if kind not in {"resolve", "clarify"}:
        kind = "clarify" if "question" in value and "intent" not in value else "resolve"
    if kind == "clarify":
        command = value.get("command")
        command = command.strip().casefold() if isinstance(command, str) else command
        if command not in {"look", "talk", "content", "action"}:
            command = "look"
        targets = value.get("target_keys", value.get("target_key"))
        if isinstance(targets, str):
            targets = [targets]
        targets = [item.strip() for item in _coerce_list(targets) if isinstance(item, str) and item.strip()]
        return {
            "kind": "clarify",
            "question": _coerce_text(value.get("question")),
            "command": command,
            "target_keys": list(dict.fromkeys(targets))[:8],
        }
    out: dict[str, Any] = {"kind": "resolve"}
    out.update({key: value[key] for key in _RESOLUTION_FIELDS if key in value})
    for key in ("speech_text", "action_text", "narration", "character_event"):
        out[key] = _coerce_text(out.get(key))
    intent = out.get("intent")
    intent = intent.strip().casefold() if isinstance(intent, str) else intent
    if intent not in {"speech", "action", "mixed", "observation"}:
        has_speech = isinstance(out["speech_text"], str) and bool(out["speech_text"].strip())
        has_action = isinstance(out["action_text"], str) and bool(out["action_text"].strip())
        intent = "mixed" if has_speech and has_action else "speech" if has_speech else "action"
    out["intent"] = intent
    out["target_key"] = _coerce_optional_text(out.get("target_key"))
    out["hook"] = _coerce_optional_text(out.get("hook"))
    out["action_succeeded"] = _coerce_bool(out.get("action_succeeded"), True)
    out["character_should_reply"] = _coerce_bool(out.get("character_should_reply"), True)
    out["important_facts"] = [fact for fact in (_coerce_fact(raw) for raw in _coerce_list(value.get("important_facts"))) if fact]
    out["entity_additions"] = [entity for entity in (_coerce_entity(raw) for raw in _coerce_list(value.get("entity_additions"))) if entity]
    return out


def _coerce_choice(raw: Any) -> Any:
    if isinstance(raw, str):
        text = raw.strip()
        quoted = (text.startswith("「") and text.endswith("」")) or (text.startswith('"') and text.endswith('"'))
        return {"kind": "speech" if quoted else "action", "text": text.strip("「」\"")}
    if not isinstance(raw, dict):
        return raw
    if "steps" in raw:
        return raw
    step = {key: raw[key] for key in ("kind", "text", "target_key") if key in raw}
    if "text" not in step:
        for alias in ("label", "content", "line", "speech", "action"):
            if alias in raw:
                step["text"] = raw[alias]
                if alias == "speech" and "kind" not in step:
                    step["kind"] = "speech"
                if alias == "action" and "kind" not in step:
                    step["kind"] = "action"
                break
    kind = step.get("kind", raw.get("type"))
    if isinstance(kind, str):
        kind = kind.strip().casefold()
        kind = "speech" if kind in _SPEECH_WORDS else "action" if kind in _ACTION_WORDS else kind
    step["kind"] = kind
    step["text"] = _coerce_text(step.get("text"))
    if step["kind"] == "speech" and isinstance(step["text"], str):
        step["text"] = step["text"].strip().strip("「」")
    target = _coerce_optional_text(step.get("target_key"))
    if target is None:
        step.pop("target_key", None)
    else:
        step["target_key"] = target
    return step


def _coerce_choices(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        value = {"choices": value}
    if not isinstance(value, dict):
        return {"choices": []}
    choices = value.get("choices")
    if choices is None:
        for alias in ("options", "candidates", "suggestions"):
            if alias in value:
                choices = value[alias]
                break
    return {"choices": [_coerce_choice(raw) for raw in _coerce_list(choices)]}


def _coerce_summary(value: dict[str, Any]) -> dict[str, Any]:
    ids = _coerce_list(value.get("source_event_ids", value.get("event_ids")))
    return {
        "text": _coerce_text(value.get("text", value.get("summary"))),
        "source_event_ids": [item for item in ids if isinstance(item, str)],
    }


def coerce_model_output(role: str, value: Any) -> Any:
    """Drop unknown fields and repair trivially wrong types before strict checks.

    This never invents scene content; it only maps obvious variants (string
    booleans, null strings, unknown keys, alias names) onto the contract.
    """
    if role == "choices":
        return _coerce_choices(value)
    if not isinstance(value, dict):
        return value
    if role == "opening":
        return _coerce_opening(value)
    if role == "resolve":
        return _coerce_resolution(value)
    if role == "summary":
        return _coerce_summary(value)
    return value
