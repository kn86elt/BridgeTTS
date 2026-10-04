from __future__ import annotations

import json
import threading
from typing import Any


class FakeGameLLM:
    def __init__(self):
        self.opening_calls = 0
        self.resolve_calls: list[dict[str, Any]] = []
        self.character_contexts: list[dict[str, Any]] = []
        self.choice_calls: list[dict[str, Any]] = []
        self.fail_openings = 0
        self.opening_entered: threading.Event | None = None
        self.opening_release: threading.Event | None = None
        self.fail_next_choices = False
        self.resolve_entered: threading.Event | None = None
        self.resolve_release: threading.Event | None = None

    def snapshot(self):
        return self

    def opening(self, seed):
        self.opening_calls += 1
        if self.opening_entered:
            self.opening_entered.set()
            if self.opening_release:
                self.opening_release.wait(5)
        if self.fail_openings:
            self.fail_openings -= 1
            raise RuntimeError("fake opening failure")
        return {
            "location": "小さな部屋",
            "scene_time": "夕方",
            "narration": "夕方の部屋に窓と秘密の扉が見えます。ミオは窓辺にいます。",
            "character_should_reply": True,
            "character_event": "窓から風が通り、ミオにも届きます。",
            "entities": [
                {
                    "key": "window", "name": "窓", "kind": "object",
                    "perceived_by": ["player", "character"],
                    "description_by_view": {"player": "少し開いている", "character": "風の音がする"},
                },
                {
                    "key": "secret_door", "name": "秘密の扉", "kind": "object",
                    "perceived_by": ["player"],
                    "description_by_view": {"player": "壁に隠れている", "character": "この説明は読めない"},
                },
            ],
            "important_facts": [
                {"text": "壁の秘密の扉を見つけた", "known_by": ["player"]},
                {"text": "窓から風が入る", "known_by": ["player", "character"]},
            ],
        }

    def character_reply(self, payload):
        context = json.loads(payload["messages"][-1]["content"])
        self.character_contexts.append(context)
        return "窓から風が来るね。"

    def choices(self, payload):
        self.choice_calls.append(payload)
        if self.fail_next_choices:
            self.fail_next_choices = False
            raise RuntimeError("fake choices failure")
        if any(entity.get("key") == "new_key" for entity in payload.get("player_context", {}).get("scene", {}).get("entities", [])):
            return {"choices": [
                {"steps": [{"kind": "action", "text": "錆びた鍵を見る", "target_key": "new_key"}]},
                {"steps": [{"kind": "speech", "text": "鍵を見つけたよ", "target_key": "character"}]},
                {"steps": [{"kind": "action", "text": "その場で待つ"}]},
            ]}
        if payload.get("kind") == "content":
            lines = ["今日はいい天気だね。", "窓の音が気になるね。", "一緒に休もう。"]
            return {"choices": [{"steps": [{"kind": "speech", "text": line}]} for line in lines]}
        return {"choices": [
            {"steps": [{"kind": "speech", "text": "そうなんだね。", "target_key": "character"}]},
            {"steps": [{"kind": "action", "text": "窓を見る", "target_key": "window"}]},
            {"steps": [{"kind": "action", "text": "椅子に座る"}]},
        ]}

    def resolve(self, payload):
        self.resolve_calls.append(payload)
        if payload.get("input") == "EVENT_ADD":
            return {
                "kind": "resolve", "intent": "action", "speech_text": "", "action_text": "EVENT_ADD",
                "target_key": None, "narration": "軒先に錆びた鍵と秘密の箱が落ちています。",
                "action_succeeded": True, "character_should_reply": True,
                "character_event": "足元に錆びた鍵が落ちる音を聞いて、驚いて振り返ります。",
                "hook": None, "important_facts": [],
                "entity_additions": [
                    {"key": "new_key", "name": "錆びた鍵", "kind": "object",
                     "perceived_by": ["player", "character"],
                     "description_by_view": {"player": "軒先に落ちた小さな鍵", "character": "足元に落ちた金属音のする鍵"}},
                    {"key": "private_box", "name": "秘密の箱", "kind": "object",
                     "perceived_by": ["player"],
                     "description_by_view": {"player": "蓋の隙間が暗い箱"}},
                ],
            }
        if self.resolve_entered:
            self.resolve_entered.set()
            if self.resolve_release:
                self.resolve_release.wait(5)
        if payload.get("input") == "窓を見て":
            return {"kind": "clarify", "question": "どの窓を見ますか？", "command": "look", "target_keys": ["window"]}
        if payload.get("input") == "直接の発話。":
            return {
                "kind": "resolve", "intent": "speech", "speech_text": "勝手に変えられた文。",
                "action_text": "", "target_key": "character", "narration": "", "action_succeeded": False,
                "character_should_reply": True, "character_event": "", "hook": None, "important_facts": [],
            }
        selection = payload.get("selection") or {}
        selected_steps = selection.get("steps", [])
        return {
            "kind": "resolve", "intent": "action", "speech_text": "", "action_text": "",
            "target_key": None, "narration": "行動の結果、窓辺で小さな音がしました。",
            "action_succeeded": True, "character_should_reply": True,
            "character_event": "窓辺で音がして、ミオにも聞こえました。" if selected_steps else "風の音がミオにも届きます。",
            "hook": None,
            "important_facts": [{"text": "秘密の扉はプレイヤーだけが知っている", "known_by": ["player"]}],
        }

    def summarize(self, payload):
        event_ids = [event["event_id"] for turn in payload["turns"] for event in turn["events"]]
        return {"text": "窓辺での出来事を覚えている。", "source_event_ids": event_ids}
