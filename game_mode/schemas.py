"""JSON schemas sent as ``response_format`` to backends that support it.

They mirror the structural contract in validation.py but stay permissive where
small models struggle; strict validation still runs on the decoded value.
"""

from __future__ import annotations

from typing import Any

_VIEWS = {"type": "array", "items": {"type": "string", "enum": ["player", "character"]}}

_FACT = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "known_by": _VIEWS},
    "required": ["text", "known_by"],
    "additionalProperties": False,
}

_ENTITY = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "name": {"type": "string"},
        "kind": {"type": "string", "enum": ["object", "person", "place"]},
        "perceived_by": _VIEWS,
        "description_by_view": {
            "type": "object",
            "properties": {"player": {"type": "string"}, "character": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    "required": ["name", "kind", "perceived_by", "description_by_view"],
    "additionalProperties": False,
}

OPENING = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "scene_time": {"type": "string"},
        "narration": {"type": "string"},
        "character_should_reply": {"type": "boolean"},
        "character_event": {"type": "string"},
        "entities": {"type": "array", "items": _ENTITY},
        "important_facts": {"type": "array", "items": _FACT},
    },
    "required": ["location", "narration", "character_should_reply", "character_event"],
    "additionalProperties": False,
}

_RESOLVE_BRANCH = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["resolve"]},
        "intent": {"type": "string", "enum": ["speech", "action", "mixed", "observation"]},
        "speech_text": {"type": "string"},
        "action_text": {"type": "string"},
        "target_key": {"type": ["string", "null"]},
        "narration": {"type": "string"},
        "action_succeeded": {"type": "boolean"},
        "character_should_reply": {"type": "boolean"},
        "character_event": {"type": "string"},
        "hook": {"type": ["string", "null"]},
        "important_facts": {"type": "array", "items": _FACT},
        "entity_additions": {"type": "array", "items": _ENTITY},
    },
    "required": ["kind", "intent", "narration", "action_succeeded", "character_should_reply", "character_event"],
    "additionalProperties": False,
}

_CLARIFY_BRANCH = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["clarify"]},
        "question": {"type": "string"},
        "command": {"type": "string", "enum": ["look", "talk", "content", "action"]},
        "target_keys": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "question", "command", "target_keys"],
    "additionalProperties": False,
}

RESOLVE = {"anyOf": [_RESOLVE_BRANCH, _CLARIFY_BRANCH]}

CHOICES = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["speech", "action"]},
                    "text": {"type": "string"},
                    "target_key": {"type": ["string", "null"]},
                },
                "required": ["kind", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["choices"],
    "additionalProperties": False,
}

SUMMARY = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "source_event_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "source_event_ids"],
    "additionalProperties": False,
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "opening": OPENING,
    "resolve": RESOLVE,
    "choices": CHOICES,
    "summary": SUMMARY,
}


def response_format(role: str, mode: str | None) -> dict[str, Any] | None:
    """Build the ``response_format`` argument for ``mode`` or ``None`` for plain text."""
    if mode is None or role not in SCHEMAS:
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {"type": "json_schema", "json_schema": {"name": f"game_{role}", "schema": SCHEMAS[role]}}
