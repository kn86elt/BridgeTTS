from __future__ import annotations

import threading
import unittest

from game_mode import GameEngine, GameError
from game_mode.memory import add_turn, compact_if_needed, new_memory, view_context
from game_mode.transcript import GameTranscriptWriter
from tests.fakes import FakeGameLLM


class GameEngineTests(unittest.TestCase):
    def setUp(self):
        self.llm = FakeGameLLM()
        self.epoch = "epoch-1"
        self.engine = GameEngine(
            self.llm,
            conversation_state=lambda: (self.epoch, 0),
            transcript_save=lambda _session: None,
        )
        self.page_id = "page-1"
        self.started = self.engine.start(
            page_id=self.page_id,
            conversation_epoch=self.epoch,
            normal_revision=0,
            request_id="start-1",
            seed={
                "character_id": "mio", "character_name": "ミオ",
                "system_prompt": "ミオの人物設定", "normal_history": [{"role": "user", "content": "いつもの部屋だね"}],
                "voice": {"tts_available": False},
            },
        )
        self.game_id = self.started["game_id"]

    def submit_text(self, text, *, view=None, request_id=None, revision=None, interaction_id=None):
        current = view or self.engine.view(self.game_id, self.page_id)
        return self.engine.submit(
            self.game_id,
            page_id=self.page_id,
            conversation_epoch=self.epoch,
            revision=current["revision"] if revision is None else revision,
            request_id=request_id or f"req-{text}-{len(self.llm.resolve_calls)}-{current['revision']}",
            request_input={"kind": "text", "text": text, "interaction_id": interaction_id or current["interaction"]["id"]},
        )

    def submit_choice(self, choice, *, view=None, request_id="choice-request"):
        current = view or self.engine.view(self.game_id, self.page_id)
        return self.engine.submit(
            self.game_id,
            page_id=self.page_id,
            conversation_epoch=self.epoch,
            revision=current["revision"],
            request_id=request_id,
            request_input={"kind": "choice", "choice_id": choice["id"], "interaction_id": current["interaction"]["id"]},
        )

    def test_start_is_public_only_and_character_gets_perception_projection(self):
        self.assertEqual([entry["type"] for entry in self.started["transcript"]], ["narration", "character", "interaction"])
        self.assertNotIn("秘密の扉", str(self.llm.character_contexts[0]))
        self.assertNotIn("この説明は読めない", str(self.llm.character_contexts[0]))
        self.assertNotIn("hooks", self.llm.character_contexts[0]["scene"])
        self.assertNotIn("perceived_by", str(self.llm.character_contexts[0]))
        self.assertNotIn("choice", str(self.llm.character_contexts[0]).lower())
        self.assertNotIn("interaction", str(self.llm.character_contexts[0]).lower())
        self.assertTrue(all("selection" not in event for event in self.started["transcript"]))
        self.assertTrue(all("number" in choice and "selection" not in choice for choice in self.started["interaction"]["choices"]))

    def test_view_context_filters_entity_metadata_and_private_facts(self):
        memory = new_memory({
            "location": "room", "scene_time": "night", "hooks": [{"status": "open", "topic_key": "secret"}],
            "last_hook_world_turn": 8, "entities": [
                {"id": "secret-id", "key": "secret-key", "name": "隠し扉", "kind": "object",
                 "perceived_by": ["player"], "description_by_view": {"player": "見える", "character": "漏れてはいけない説明"}},
            ],
        })
        memory["important_facts"]["player"].append({"text": "秘密の内容", "source_event_ids": ["event-secret"]})
        memory["important_facts"]["character"].append({"text": "知覚した音", "source_event_ids": ["event-sound"]})
        memory["recent_turns"].append({"world_turn": 2, "by_view": {
            "player": [{"event_id": "event-secret", "type": "action", "text": "秘密の行動"}],
            "character": [{"event_id": "event-sound", "type": "environment", "text": "音がした"}],
        }})
        context = view_context(memory, "character")
        serialized = str(context)
        for hidden in ("隠し扉", "secret-id", "secret-key", "漏れてはいけない説明", "topic_key", "last_hook_world_turn", "秘密の内容", "event-secret"):
            self.assertNotIn(hidden, serialized)
        self.assertIn("音がした", serialized)

    def test_fullwidth_numeric_choice_resolves_server_selection_once(self):
        initial = self.engine.view(self.game_id, self.page_id)
        expected_label = initial["interaction"]["choices"][1]["label"]
        request_id = "numeric-choice"
        result = self.submit_text("２", request_id=request_id)
        turn_after = result["world_turn"]
        self.assertEqual(turn_after, 2)
        self.assertTrue(any(event["type"] == "player" and event["text"] == expected_label for event in result["events"]))
        self.assertFalse(any(event["type"] == "player" and event["text"] == "２" for event in result["events"]))
        calls = len(self.llm.resolve_calls)
        replay = self.submit_text("２", request_id=request_id, view=initial)
        self.assertEqual(replay, result)
        self.assertEqual(len(self.llm.resolve_calls), calls)
        self.assertEqual(self.engine.view(self.game_id, self.page_id)["world_turn"], turn_after)
        with self.assertRaises(GameError) as reused:
            self.submit_text("別の入力", request_id=request_id, view=initial)
        self.assertEqual(reused.exception.code, "REQUEST_ID_REUSED")
        self.assertNotIn("transcript", result)
        stored = self.engine.request_result(self.game_id, self.page_id, request_id)
        self.assertEqual(stored["result"], result)
        response_ref = self.engine.current(self.page_id).request_records[request_id]["response"]
        self.assertIn("_event_ids", response_ref)
        self.assertNotIn("events", response_ref)

    def test_look_confirmation_stale_choice_is_rejected_without_world_update(self):
        initial = self.engine.view(self.game_id, self.page_id)
        clarification = self.submit_text("見る", request_id="look-command")
        with self.assertRaises(GameError) as error:
            self.engine.submit(
                self.game_id, page_id=self.page_id, conversation_epoch=self.epoch,
                revision=clarification["revision"], request_id="stale-interaction",
                request_input={"kind": "choice", "choice_id": initial["interaction"]["choices"][0]["id"], "interaction_id": initial["interaction"]["id"]},
            )
        self.assertEqual(error.exception.code, "STALE_REVISION")
        after = self.engine.view(self.game_id, self.page_id)
        self.assertEqual(after["world_turn"], initial["world_turn"])
        self.assertEqual(after["revision"], clarification["revision"])

        hidden = next(choice for choice in clarification["interaction"]["choices"] if choice["label"] == "秘密の扉")
        result = self.submit_choice(hidden, view=clarification, request_id="look-hidden-door")
        self.assertEqual(result["world_turn"], initial["world_turn"] + 1)
        self.assertNotIn("秘密の扉", str(self.llm.character_contexts[-1]))
        self.assertNotIn("秘密の内容", str(self.llm.character_contexts[-1]))
        self.assertNotIn("秘密の行動", str(self.llm.character_contexts[-1]))

    def test_話す_requires_target_then_user_selected_content(self):
        initial = self.engine.view(self.game_id, self.page_id)
        target = self.submit_text("話す", request_id="talk-command")
        self.assertEqual(target["world_turn"], initial["world_turn"])
        self.assertEqual(target["interaction"]["kind"], "target")
        self.assertEqual([choice["label"] for choice in target["interaction"]["choices"]], ["ミオ"])
        content = self.submit_choice(target["interaction"]["choices"][0], view=target, request_id="talk-target")
        self.assertEqual(content["world_turn"], initial["world_turn"])
        self.assertEqual(content["interaction"]["kind"], "content")
        result = self.submit_choice(content["interaction"]["choices"][0], view=content, request_id="talk-content")
        self.assertEqual(result["world_turn"], initial["world_turn"] + 1)
        self.assertTrue(any(entry["type"] == "player" and "今日はいい天気" in entry["text"] for entry in result["events"]))

    def test_direct_speech_is_preserved_verbatim(self):
        result = self.submit_text("直接の発話。", request_id="direct-speech")
        self.assertTrue(any(entry["type"] == "player" and entry["text"] == "直接の発話。" for entry in result["events"]))
        self.assertIn("直接の発話。", str(self.llm.character_contexts[-1]))
        self.assertEqual(result["world_turn"], self.started["world_turn"] + 1)

    def test_character_can_distinguish_its_previous_reply_from_selected_speech(self):
        initial = self.engine.view(self.game_id, self.page_id)
        self.submit_choice(initial["interaction"]["choices"][0], view=initial)
        context = self.llm.character_contexts[-1]
        events = [event for turn in context["recent_turns"] for event in turn["events"]]
        self.assertTrue(any(event.get("speaker") == "character" and event["text"] == "窓から風が来るね。" for event in events))
        self.assertTrue(any(event.get("speaker") == "player" and event["text"] == "そうなんだね。" for event in events))
        self.assertEqual(context["current_turn"]["world_turn"], initial["world_turn"] + 1)
        self.assertTrue(any(event.get("speaker") == "player" and event["text"] == "そうなんだね。"
                            for event in context["current_turn"]["events"]))

    def test_gm_adds_only_announced_entities_to_viewpoint_scene(self):
        before = self.engine.view(self.game_id, self.page_id)
        result = self.submit_text("EVENT_ADD", request_id="add-entity", view=before)
        self.assertEqual(result["world_turn"], before["world_turn"] + 1)
        self.assertIn("錆びた鍵を見る", [choice["label"] for choice in result["interaction"]["choices"]])
        self.assertNotIn("selection", result["interaction"]["choices"][0])
        session = self.engine.current(self.page_id)
        new_entity = next(entity for entity in session.scene["entities"] if entity["key"] == "new_key")
        self.assertEqual(session.pending.choices[0].selection["steps"][0]["target_id"], new_entity["id"])
        character_context = self.llm.character_contexts[-1]
        names = [entity["name"] for entity in character_context["scene"]["entities"]]
        self.assertIn("錆びた鍵", names)
        self.assertNotIn("秘密の箱", names)
        self.assertNotIn("private_box", str(character_context))

    def test_transcript_contains_only_public_display_entries(self):
        session = self.engine.current(self.page_id)
        encoded = GameTranscriptWriter().encode(session).decode("utf-8")
        self.assertIn('"format": "bridgetts.game.transcript"', encoded)
        self.assertIn("秘密の扉", encoded)  # It was shown in player-facing narration.
        for internal in ("description_by_view", "source_event_ids", "request_records", "player_only", "seed", "hooks"):
            self.assertNotIn(internal, encoded)

    def test_audio_is_character_only_and_idempotent_by_request(self):
        engine = GameEngine(FakeGameLLM(), conversation_state=lambda: ("epoch", 0))
        started = engine.start(
            page_id="audio-page", conversation_epoch="epoch", normal_revision=0,
            request_id="audio-start", seed={"character_name": "ミオ", "voice": {"tts_available": True, "voice_path": "voice.wav"}},
        )
        character_event = next(event for event in started["transcript"] if event["type"] == "character")
        first = engine.audio_payload(started["game_id"], "audio-page", character_event["id"], character_event["revision"], "audio-request")
        replay = engine.audio_payload(started["game_id"], "audio-page", character_event["id"], character_event["revision"], "audio-request")
        self.assertEqual(first["audio_req_id"], replay["audio_req_id"])
        self.assertTrue(replay["already_started"])
        duplicate = engine.audio_payload(started["game_id"], "audio-page", character_event["id"], character_event["revision"], "second-request")
        self.assertTrue(duplicate["already_started"])
        self.assertEqual(duplicate["audio_req_id"], first["audio_req_id"])
        retry = engine.audio_payload(started["game_id"], "audio-page", character_event["id"], character_event["revision"], "audio-retry", retry=True)
        self.assertNotEqual(retry["audio_req_id"], first["audio_req_id"])
        narration = next(event for event in started["transcript"] if event["type"] == "narration")
        with self.assertRaises(GameError) as invalid:
            engine.audio_payload(started["game_id"], "audio-page", narration["id"], narration["revision"], "narration-audio")
        self.assertEqual(invalid.exception.code, "INVALID_AUDIO_EVENT")

    def test_generation_failure_keeps_original_state_and_failed_id_is_not_replayed(self):
        before = self.engine.view(self.game_id, self.page_id)
        def fail_resolution(payload):
            self.llm.resolve_calls.append(payload)
            raise RuntimeError("resolution failed")
        self.llm.resolve = fail_resolution
        request = {
            "kind": "text", "text": "自由な行動", "interaction_id": before["interaction"]["id"],
        }
        args = dict(page_id=self.page_id, conversation_epoch=self.epoch, revision=before["revision"], request_input=request)
        with self.assertRaises(GameError) as first:
            self.engine.submit(self.game_id, request_id="fails-once", **args)
        self.assertEqual(first.exception.code, "INVALID_MODEL_OUTPUT")
        self.assertEqual(self.engine.view(self.game_id, self.page_id), before)
        resolve_calls = len(self.llm.resolve_calls)
        choice_calls = len(self.llm.choice_calls)
        with self.assertRaises(GameError) as replay:
            self.engine.submit(self.game_id, request_id="fails-once", **args)
        self.assertEqual(replay.exception.code, "INVALID_MODEL_OUTPUT")
        self.assertEqual(len(self.llm.resolve_calls), resolve_calls)
        self.assertEqual(len(self.llm.choice_calls), choice_calls)

    def test_choice_failure_commits_turn_and_exact_replay_does_not_generate_again(self):
        before = self.engine.view(self.game_id, self.page_id)
        self.llm.fail_next_choices = True
        result = self.submit_text("直接の発話。", view=before, request_id="choice-fallback")
        self.assertEqual(result["world_turn"], before["world_turn"] + 1)
        self.assertTrue(any(event["type"] == "player" and event["text"] == "直接の発話。" for event in result["events"]))
        self.assertEqual(len(result["interaction"]["choices"]), 3)
        self.assertTrue(result["notice"])
        calls = len(self.llm.resolve_calls)
        replay = self.submit_text("直接の発話。", view=before, request_id="choice-fallback")
        self.assertEqual(replay, result)
        self.assertEqual(len(self.llm.resolve_calls), calls)
        choice = result["interaction"]["choices"][0]
        next_result = self.submit_choice(choice, view=result, request_id="use-fallback")
        self.assertEqual(next_result["world_turn"], result["world_turn"] + 1)
        self.assertIsNone(next_result["notice"])

    def test_start_tolerates_unintroduced_object_and_registers_character_only_once(self):
        original = self.llm.opening
        def noisy_opening(seed):
            result = original(seed)
            result["entities"].append({"key": "duplicate", "name": "ミオ", "kind": "person"})
            result["entities"].append({"key": "unknown", "name": "本文にない物"})
            result["extra"] = "optional metadata"
            return result
        self.llm.opening = noisy_opening
        self.engine.close_page(self.page_id)
        started = self.engine.start(page_id="new-page", conversation_epoch=self.epoch, normal_revision=0,
                                    request_id="tolerant-start", seed={"character_name": "ミオ"})
        session = self.engine.current("new-page")
        self.assertEqual(started["status"], "active")
        self.assertEqual(sum(entity["name"] == "ミオ" for entity in session.scene["entities"]), 1)
        self.assertNotIn("本文にない物", str(session.scene))

    def test_start_failure_replay_does_not_call_fake_model_twice(self):
        failing = FakeGameLLM()
        failing.fail_openings = 1
        engine = GameEngine(failing, conversation_state=lambda: ("epoch", 0))
        args = dict(page_id="p", conversation_epoch="epoch", normal_revision=0, seed={"character_name": "ミオ"})
        with self.assertRaises(GameError):
            engine.start(request_id="start-fail", **args)
        with self.assertRaises(GameError):
            engine.start(request_id="start-fail", **args)
        self.assertEqual(failing.opening_calls, 1)
        engine.start(request_id="start-retry", **args)
        self.assertEqual(failing.opening_calls, 2)

    def test_same_running_request_returns_request_running_and_other_input_is_busy(self):
        entered = threading.Event()
        release = threading.Event()
        self.llm.resolve_entered = entered
        self.llm.resolve_release = release
        before = self.engine.view(self.game_id, self.page_id)
        request_input = {"kind": "text", "text": "自由な行動", "interaction_id": before["interaction"]["id"]}
        kwargs = dict(page_id=self.page_id, conversation_epoch=self.epoch, revision=before["revision"], request_input=request_input)
        errors: list[GameError] = []

        def run():
            try:
                self.engine.submit(self.game_id, request_id="same-in-flight", **kwargs)
            except GameError as exc:
                errors.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(entered.wait(2))
        with self.assertRaises(GameError) as retry:
            self.engine.submit(self.game_id, request_id="same-in-flight", **kwargs)
        self.assertEqual(retry.exception.code, "REQUEST_RUNNING")
        with self.assertRaises(GameError) as concurrent:
            self.engine.submit(self.game_id, request_id="different-request", **kwargs)
        self.assertEqual(concurrent.exception.code, "GAME_BUSY")
        release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.engine.view(self.game_id, self.page_id)["world_turn"], before["world_turn"] + 1)

    def test_suspend_resume_invalidates_pending_generation_and_preserves_game(self):
        entered = threading.Event()
        release = threading.Event()
        self.llm.resolve_entered = entered
        self.llm.resolve_release = release
        before = self.engine.view(self.game_id, self.page_id)
        request_input = {"kind": "text", "text": "自由な行動", "interaction_id": before["interaction"]["id"]}
        errors: list[GameError] = []

        def run():
            try:
                self.engine.submit(self.game_id, page_id=self.page_id, conversation_epoch=self.epoch,
                                   revision=before["revision"], request_id="pending-turn", request_input=request_input)
            except GameError as exc:
                errors.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(entered.wait(2))
        suspended = self.engine.suspend(self.game_id, page_id=self.page_id, revision=before["revision"], request_id="suspend")
        self.assertEqual(suspended["status"], "suspended")
        release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.engine.view(self.game_id, self.page_id)["world_turn"], before["world_turn"])
        resumed = self.engine.resume(self.game_id, page_id=self.page_id, conversation_epoch=self.epoch,
                                     revision=suspended["revision"], request_id="resume")
        self.assertEqual(resumed["status"], "active")
        self.assertEqual(resumed["world_turn"], before["world_turn"])
        self.assertEqual(resumed["revision"], suspended["revision"] + 1)

    def test_summary_generation_is_separate_for_each_view(self):
        memory = new_memory({"location": "room", "scene_time": "", "entities": [], "hooks": []})
        for turn in range(1, 14):
            add_turn(memory, turn, [{
                "id": f"event-{turn}", "world_turn": turn, "kind": "environment", "important": False,
                "text_by_view": {"player": "壁の秘密の記号" if turn == 1 else "部屋の様子", "character": "窓から風が吹く"},
            }])
        received = {}

        def summarize(view, payload):
            received[view] = payload
            ids = [event["event_id"] for turn in payload["turns"] for event in turn["events"]]
            return {"text": f"{view}視点の要約", "source_event_ids": ids}

        compact_if_needed(memory, 13, summarize)
        self.assertIn("壁の秘密の記号", str(received["player"]))
        self.assertNotIn("壁の秘密の記号", str(received["character"]))
        self.assertIn("窓から風が吹く", str(received["character"]))
        self.assertEqual(memory["summaries"]["character"], "character視点の要約")

    def test_close_page_during_start_discards_late_opening(self):
        fake = FakeGameLLM()
        fake.opening_entered = threading.Event()
        fake.opening_release = threading.Event()
        engine = GameEngine(fake, conversation_state=lambda: ("epoch", 0))
        failures: list[GameError] = []

        def start():
            try:
                engine.start(page_id="page", conversation_epoch="epoch", normal_revision=0,
                             request_id="late-start", seed={"character_name": "ミオ"})
            except GameError as exc:
                failures.append(exc)

        worker = threading.Thread(target=start)
        worker.start()
        self.assertTrue(fake.opening_entered.wait(2))
        engine.close_page("page")
        fake.opening_release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].code, "GAME_CLOSED")
        self.assertIsNone(engine.current("page"))
        with self.assertRaises(GameError) as replay:
            engine.start(page_id="page", conversation_epoch="epoch", normal_revision=0,
                         request_id="late-start", seed={"character_name": "ミオ"})
        self.assertEqual(replay.exception.code, "GAME_CLOSED")
        self.assertEqual(fake.opening_calls, 1)

    def test_transition_after_suspend_closes_state_and_prevents_late_audio(self):
        audio_llm = FakeGameLLM()
        engine = GameEngine(audio_llm, conversation_state=lambda: ("epoch", 0))
        started = engine.start(
            page_id="page", conversation_epoch="epoch", normal_revision=0,
            request_id="start", seed={"character_name": "ミオ", "voice": {"tts_available": True}},
        )
        char_event = next(event for event in started["transcript"] if event["type"] == "character")
        suspended = engine.suspend(started["game_id"], page_id="page", revision=started["revision"], request_id="suspend")
        engine.transition("page", "epoch")
        self.assertIsNone(engine.current("page"))
        with self.assertRaises(GameError) as audio:
            engine.audio_payload(started["game_id"], "page", char_event["id"], char_event["revision"], "late-audio")
        self.assertEqual(audio.exception.code, "GAME_CLOSED")
        with self.assertRaises(GameError) as resume:
            engine.resume(started["game_id"], page_id="page", conversation_epoch="epoch",
                          revision=suspended["revision"], request_id="late-resume")
        self.assertEqual(resume.exception.code, "GAME_CLOSED")


if __name__ == "__main__":
    unittest.main()
