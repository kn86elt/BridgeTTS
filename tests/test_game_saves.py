from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from game_mode import GameEngine, GameError
from game_mode.saves import GameSaveStore
from tests.fakes import FakeGameLLM


class GameSaveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = GameSaveStore(self.directory.name)
        self.llm = FakeGameLLM()
        self.engine = GameEngine(self.llm, conversation_state=lambda: ("epoch", 0))
        self.view = self.engine.start(page_id="page", conversation_epoch="epoch", normal_revision=0,
                                      request_id="start", seed={"character_id": "mio", "character_name": "ミオ",
                                      "system_prompt": "秘密の設定", "normal_history": [],
                                      "voice": {"tts_available": True, "voice_path": "voice.wav"}})

    def save(self, request_id="save", name="夜の部屋"):
        return self.engine.save(self.store, self.view["game_id"], page_id="page", revision=self.view["revision"],
                                request_id=request_id, name=name)

    def load(self, saved, engine=None, **overrides):
        options = dict(page_id="page", conversation_epoch="epoch", normal_revision=0, request_id="load",
                       expected_game_id=self.view["game_id"], expected_revision=self.view["revision"])
        options.update(overrides)
        return (engine or self.engine).load(self.store, saved["id"], **options)

    def submit(self, text):
        self.view = self.engine.submit(self.view["game_id"], page_id="page", conversation_epoch="epoch",
                                       revision=self.view["revision"], request_id=f"turn-{self.view['revision']}",
                                       request_input={"kind": "text", "text": text,
                                                      "interaction_id": self.view["interaction"]["id"]})

    def assert_error(self, code, function, *args, **kwargs):
        with self.assertRaises(GameError) as raised:
            function(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)

    def test_restart_restores_private_state_and_continues_without_new_opening(self):
        for _ in range(14):
            self.submit("こんにちは")
        original = self.engine.current("page")
        self.assertTrue(original.memory["summaries"]["player"])
        saved = self.save()
        restarted = GameEngine(FakeGameLLM(), conversation_state=lambda: ("new-epoch", 9))
        loaded = self.load(saved, restarted, page_id="new-page", conversation_epoch="new-epoch",
                           normal_revision=9, expected_game_id=None, expected_revision=None)
        restored = restarted.current("new-page")
        self.assertNotEqual(loaded["game_id"], original.game_id)
        for key in ("seed", "scene", "memory", "base_choices", "public_entries", "world_turn", "operation_count"):
            self.assertEqual(getattr(restored, key), getattr(original, key), key)
        self.assertFalse(restored.request_records)
        self.assertFalse(restored.audio_records)
        self.assertEqual(restarted.llm.opening_calls, 0)
        self.assertNotIn("秘密の設定", str(loaded))
        next_view = restarted.submit(loaded["game_id"], page_id="new-page", conversation_epoch="new-epoch",
                                     revision=loaded["revision"], request_id="next",
                                     request_input={"kind": "choice", "choice_id": loaded["interaction"]["choices"][0]["id"],
                                                    "interaction_id": loaded["interaction"]["id"]})
        self.assertEqual(next_view["world_turn"], original.world_turn + 1)
        character_event = next(e for e in loaded["transcript"] if e["type"] == "character")
        audio = restarted.audio_payload(loaded["game_id"], "new-page", character_event["id"], character_event["revision"], "audio")
        self.assertFalse(audio["already_started"])

    def test_pending_target_and_cancel_survive_load(self):
        self.submit("見る")
        self.assertEqual(self.view["interaction"]["kind"], "target")
        saved = self.save()
        original = self.engine.current("page")
        self.view = self.load(saved)
        restored = self.engine.current("page")
        self.assertEqual(restored.pending.choices, original.pending.choices)
        self.assertEqual(restored.pending.command, original.pending.command)
        self.assertNotEqual(restored.pending.id, original.pending.id)
        self.submit("キャンセル")
        self.assertEqual(self.view["world_turn"], original.world_turn)
        self.assertEqual(self.engine.current("page").pending.choices, original.base_choices)

    def test_save_retry_is_one_file_and_load_retry_does_not_rewind(self):
        saved = self.save()
        self.assertEqual(self.save(), saved)
        self.assertEqual(len(self.store.list()["saves"]), 1)
        self.assert_error("REQUEST_ID_REUSED", self.save, name="別名")
        original_view = self.view
        loaded = self.load(saved)
        self.assertEqual(self.load(saved)["game_id"], loaded["game_id"])
        self.view = loaded
        self.submit("こんにちは")
        progressed = self.view
        self.view = original_view
        replay = self.load(saved)
        self.assertEqual(replay["world_turn"], progressed["world_turn"])

    def test_failed_read_or_write_does_not_change_current_game(self):
        saved = self.save()
        before = self.engine.current("page")
        with patch("game_mode.saves.os.replace", side_effect=OSError("disk full")):
            self.assert_error("SAVE_WRITE_FAILED", self.save, request_id="second")
        self.assertEqual(self.engine.current("page"), before)
        self.assertEqual(len(self.store.list()["saves"]), 1)
        self.assertFalse(list(Path(self.directory.name).glob("*.tmp")))
        path = Path(self.directory.name) / (saved["id"] + ".json")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["state"]["world_turn"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assert_error("SAVE_CORRUPT", self.load, saved)
        self.assertEqual(self.engine.current("page"), before)
        self.assertEqual(self.store.list(), {"saves": [], "unavailable": 1})
        self.assert_error("SAVE_NOT_FOUND", self.load, {"id": "a" * 64})
        self.assert_error("INVALID_SAVE_ID", self.load, {"id": "../secret"})

    def test_owner_revision_busy_and_suspended_save(self):
        saved = self.save()
        self.assert_error("OWNER_MISMATCH", self.load, saved, page_id="other")
        self.assert_error("STALE_REVISION", self.load, saved, expected_revision=0)
        self.assert_error("STALE_CONVERSATION", self.load, saved, normal_revision=1)
        self.engine.normal_busy = lambda: True
        self.assert_error("GAME_BUSY", self.load, saved)
        self.engine.normal_busy = lambda: False
        self.engine._session.status = "resolving"
        self.assert_error("GAME_BUSY", self.load, saved)
        self.assert_error("GAME_BUSY", self.save, request_id="while-resolving")
        self.engine._session.status = "active"
        self.view = self.engine.suspend(self.view["game_id"], page_id="page", revision=self.view["revision"], request_id="suspend")
        suspended = self.save(request_id="suspended")
        self.assertEqual(self.load(suspended)["status"], "active")
        self.assertEqual(len(self.store.list()["saves"]), 2)


if __name__ == "__main__":
    unittest.main()
