from __future__ import annotations

import copy
import asyncio
import json
import threading
import unittest
import uuid

import bridge_server_api as api
from fastapi.testclient import TestClient

from tests.fakes import FakeGameLLM


class GameHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No lifespan context is entered: the tests exercise routes without
        # starting the desktop browser or background lease thread.
        cls.client = TestClient(api.app)

    def setUp(self):
        self.old_llm = api._game_engine.llm
        self.old_save = api._game_engine.transcript_save
        self.old_load_settings = api.load_settings
        self.old_seed = api._capture_game_seed
        self.old_selected = api.state.selected
        self.old_memory = copy.deepcopy(api.state.memory)
        self.old_pending = copy.deepcopy(api.state.mem_pending)
        self.old_user_turns = copy.deepcopy(api.state.mem_user_turns)
        self.old_session_ctx = copy.deepcopy(api.state.session_ctx)
        api.state.memory.append({"role": "user", "content": "通常会話の土台"})
        api.state.mem_pending = [{"role": "user", "content": "抽出保留中"}]
        api.state.mem_user_turns = 7
        api.state.session_ctx = {"started_at": "unchanged"}
        api._game_engine.llm = FakeGameLLM()
        api._game_engine.transcript_save = lambda _session: None
        api.load_settings = lambda: {**api.DEFAULT_SETTINGS, "log_auto_save": False, "tts_enabled": False}
        api._capture_game_seed = lambda: {
            "character_id": "mio", "character_name": "ミオ", "system_prompt": "人物設定",
            "normal_history": [{"role": "user", "content": "いつもの会話"}],
            "voice": {"tts_available": False},
        }
        api.state.selected = {"id": "mio", "name": "ミオ"}
        page = f"test-page-{uuid.uuid4().hex}"
        self.status = self.client.get("/conversation/status", headers={"X-Bridge-Page-Id": page}).json()
        self.page_id = page
        self.start_body = {
            "page_id": page,
            "conversation_epoch": self.status["conversation_epoch"],
            "normal_revision": self.status["normal_revision"],
            "request_id": f"start-{uuid.uuid4().hex}",
        }
        self.game = self.client.post("/game/start", json=self.start_body)
        self.assertEqual(self.game.status_code, 200, self.game.text)
        self.game_view = self.game.json()
        self.game_id = self.game_view["game_id"]

    def tearDown(self):
        try:
            api._game_engine.close_page(self.page_id)
        except Exception:
            pass
        api._game_engine.llm = self.old_llm
        api._game_engine.transcript_save = self.old_save
        api.load_settings = self.old_load_settings
        api._capture_game_seed = self.old_seed
        api.state.selected = self.old_selected
        api.state.memory = self.old_memory
        api.state.mem_pending = self.old_pending
        api.state.mem_user_turns = self.old_user_turns
        api.state.session_ctx = self.old_session_ctx

    def stream_events(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]

    def test_streamed_turn_commits_once_and_can_be_recovered_or_replayed(self):
        before_memory = copy.deepcopy(api.state.memory)
        body = {"page_id": self.page_id, "conversation_epoch": self.status["conversation_epoch"],
                "revision": self.game_view["revision"], "request_id": "stream-turn",
                "input": {"kind": "text", "text": "直接の発話。", "interaction_id": self.game_view["interaction"]["id"]}}
        events = self.stream_events(self.client.post(f"/game/{self.game_id}/turn?stream=true", json=body))
        self.assertEqual(events[0], {"type": "phase", "phase": "resolve"})
        self.assertEqual(events[-1]["type"], "result")
        result = events[-1]["result"]
        self.assertEqual(result["world_turn"], 2)
        self.assertEqual(api.state.memory, before_memory)
        calls = len(api._game_engine.llm.resolve_calls)
        replay = self.stream_events(self.client.post(f"/game/{self.game_id}/turn?stream=true", json=body))[-1]["result"]
        self.assertEqual(replay, result)
        self.assertEqual(len(api._game_engine.llm.resolve_calls), calls)
        recovered = self.client.get(f"/game/{self.game_id}/requests/stream-turn", params={"page_id": self.page_id}).json()
        self.assertEqual(recovered["result"], result)
        self.assertNotIn("gm_context", str(events))

    def test_streamed_failure_keeps_previous_turn_and_reports_code(self):
        body = {"page_id": self.page_id, "conversation_epoch": self.status["conversation_epoch"],
                "revision": self.game_view["revision"], "request_id": "bad-stream-choice",
                "input": {"kind": "choice", "choice_id": "nonexistent", "interaction_id": self.game_view["interaction"]["id"]}}
        events = self.stream_events(self.client.post(f"/game/{self.game_id}/turn?stream=true", json=body))
        self.assertEqual(events[-1], {"type": "error", "code": "INVALID_CHOICE", "status": 409})
        self.assertEqual(api._game_engine.view(self.game_id, self.page_id)["world_turn"], 1)

    def test_streamed_start_returns_public_final_view(self):
        api._game_engine.close_page(self.page_id)
        self.page_id = "stream-start-page-" + uuid.uuid4().hex
        body = {**self.start_body, "page_id": self.page_id, "request_id": "stream-start"}
        events = self.stream_events(self.client.post('/game/start?stream=true', json=body))
        self.assertEqual(events[0], {"type": "phase", "phase": "opening"})
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["result"]["status"], "active")
        self.assertNotIn("seed", str(events))

    def test_game_turn_is_one_json_response_and_does_not_touch_normal_memory(self):
        normal_before = {
            "memory": copy.deepcopy(api.state.memory),
            "pending": copy.deepcopy(api.state.mem_pending),
            "user_turns": copy.deepcopy(api.state.mem_user_turns),
            "session_ctx": copy.deepcopy(api.state.session_ctx),
        }
        interaction = self.game_view["interaction"]
        response = self.client.post(f"/game/{self.game_id}/turn", json={
            "page_id": self.page_id,
            "conversation_epoch": self.status["conversation_epoch"],
            "revision": self.game_view["revision"],
            "request_id": "http-direct-speech",
            "input": {"kind": "text", "text": "直接の発話。", "interaction_id": interaction["id"]},
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers.get("content-type", "").split(";")[0], "application/json")
        payload = response.json()
        self.assertEqual(payload["world_turn"], 2)
        self.assertTrue(any(event["type"] == "player" and event["text"] == "直接の発話。" for event in payload["events"]))
        for key, value in normal_before.items():
            self.assertEqual(getattr(api.state, {"memory": "memory", "pending": "mem_pending", "user_turns": "mem_user_turns", "session_ctx": "session_ctx"}[key]), value)
        for private_key in ("seed", "scene", "memory", "selection", "gm_context"):
            self.assertNotIn(private_key, payload)

    def test_game_active_blocks_ordinary_generation_and_view_reset(self):
        before = copy.deepcopy(api.state.memory)
        headers = {
            "X-Bridge-Page-Id": self.page_id,
            "X-Conversation-Epoch": self.status["conversation_epoch"],
        }
        chat = self.client.post("/chat", headers=headers, json={"message": "通常会話へ混入してはいけない", "tts_enabled": False})
        self.assertEqual(chat.status_code, 409)
        self.assertEqual(api.state.memory, before)
        auto = self.client.post("/user_reply_gen", headers=headers)
        self.assertEqual(auto.status_code, 409)
        tts = self.client.post("/tts_direct", headers=headers, json={"text": "遅延音声"})
        self.assertEqual(tts.status_code, 409)
        reset = self.client.post("/conversation/view-reset", json={
            "page_id": self.page_id,
            "conversation_epoch": self.status["conversation_epoch"],
            "normal_revision": self.status["normal_revision"],
            "request_id": "reset-while-active",
        })
        self.assertEqual(reset.status_code, 409)
        self.assertEqual(api.state.memory, before)

    def test_normal_chat_keeps_sse_when_game_is_suspended(self):
        current = api._game_engine.current(self.page_id)
        api._game_engine.suspend(
            self.game_id, page_id=self.page_id, revision=current.revision, request_id="suspend-for-normal-chat",
        )
        original_stream = api.stream_llm_with_tts

        def fake_stream(*_args, **_kwargs):
            yield "data: fake-normal-reply" + chr(10) + chr(10)

        api.stream_llm_with_tts = fake_stream
        try:
            response = self.client.post("/chat", headers={
                "X-Bridge-Page-Id": self.page_id,
                "X-Conversation-Epoch": self.status["conversation_epoch"],
            }, json={"message": "通常会話へ戻る", "tts_enabled": False})
        finally:
            api.stream_llm_with_tts = original_stream
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers.get("content-type", "").startswith("text/event-stream"))
        self.assertIn("data: fake-normal-reply", response.text)
        self.assertEqual(api.state.memory[-1], {"role": "user", "content": "通常会話へ戻る"})

    def test_extra_normal_fields_are_rejected_and_game_is_page_owned(self):
        inter = self.game_view["interaction"]
        response = self.client.post(f"/game/{self.game_id}/turn", json={
            "page_id": self.page_id,
            "conversation_epoch": self.status["conversation_epoch"],
            "revision": self.game_view["revision"],
            "request_id": "strict-request",
            "input": {"kind": "text", "text": "見る", "interaction_id": inter["id"], "memorize": True},
        })
        self.assertEqual(response.status_code, 422)
        other_page = self.client.get(f"/game/{self.game_id}", params={"page_id": "other-page"})
        self.assertEqual(other_page.status_code, 409)
        self.assertEqual(other_page.json()["detail"]["code"], "OWNER_MISMATCH")

    def test_start_retry_returns_same_game_without_opening_twice(self):
        calls = api._game_engine.llm.opening_calls
        replay = self.client.post("/game/start", json=self.start_body)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json()["game_id"], self.game_id)
        self.assertEqual(api._game_engine.llm.opening_calls, calls)

    def test_log_setting_and_manual_export_use_public_transcript_only(self):
        class Writer:
            def __init__(self):
                self.saved = []

            def save(self, session):
                self.saved.append(session.game_id)

        writer = Writer()
        old_writer = api._game_transcript_writer
        api._game_transcript_writer = writer
        try:
            current = api._game_engine.current(self.page_id)
            self.assertIsNone(api._save_game_transcript(current))  # log_auto_save is false
            self.assertEqual(writer.saved, [])
            api.load_settings = lambda: {**api.DEFAULT_SETTINGS, "log_auto_save": True, "tts_enabled": False}
            self.assertIsNone(api._save_game_transcript(current))
            self.assertEqual(writer.saved, [self.game_id])
        finally:
            api._game_transcript_writer = old_writer

        exported = self.client.post(f"/game/{self.game_id}/log/export", json={"page_id": self.page_id})
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.headers.get("content-type", "").split(";")[0], "application/json")
        document = exported.json()
        self.assertEqual(document["format"], "bridgetts.game.transcript")
        self.assertTrue(all("selection" not in entry for entry in document["entries"]))
        rejected = self.client.post("/log/load", json={"filepath": "not-a-real-file.game.json"})
        self.assertEqual(rejected.status_code, 400)

    def test_seed_captures_existing_context_without_consuming_pending_memory(self):
        old_builder = api._build_llm_messages
        old_greeting = api._greeting_instruction
        old_load_memory = api.load_memory
        api.state.selected = {
            "name": "ミオ", "prompt": "characters/mio.txt", "voice_path": "voice.wav", "has_voice": True,
        }
        api.state.memory = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "greeting"},
            {"role": "user", "content": "通常の質問"},
            {"role": "assistant", "content": "通常の返事"},
        ]
        api._build_llm_messages = lambda _settings: [{"role": "system", "content": "人物設定と関係性"}]
        api._greeting_instruction = lambda _settings: "greeting"
        api.load_memory = lambda _name: "既存の長期記憶"
        pending = copy.deepcopy(api.state.mem_pending)
        user_turns = api.state.mem_user_turns
        context = copy.deepcopy(api.state.session_ctx)
        try:
            seed = self.old_seed()
        finally:
            api._build_llm_messages = old_builder
            api._greeting_instruction = old_greeting
            api.load_memory = old_load_memory
        self.assertEqual(seed["system_prompt"], "人物設定と関係性")
        self.assertEqual(seed["long_term_memory"], "既存の長期記憶")
        self.assertEqual([item["content"] for item in seed["normal_history"]], ["通常の質問", "通常の返事"])
        self.assertEqual(api.state.mem_pending, pending)
        self.assertEqual(api.state.mem_user_turns, user_turns)
        self.assertEqual(api.state.session_ctx, context)


class GameStreamTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_is_delivered_while_generation_is_still_running(self):
        release = threading.Event()
        finished = threading.Event()
        def operation(emit):
            release.wait(5)
            emit({"type": "character_preview", "text": "返答"})
            finished.set()
            return {"revision": 2}
        response = api._game_stream_response(operation, "resolve")
        try:
            first = await asyncio.wait_for(anext(response.body_iterator), 1)
            self.assertIn('"phase": "resolve"', first)
            self.assertFalse(finished.is_set())
        finally:
            release.set()
        events = [frame async for frame in response.body_iterator]
        self.assertIn('"character_preview"', events[0])
        self.assertIn('"result"', events[-1])


if __name__ == "__main__":
    unittest.main()
