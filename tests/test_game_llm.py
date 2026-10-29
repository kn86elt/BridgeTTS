import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from game_mode.llm import OpenAIGameLLM
from game_mode.models import GameError
from game_mode.validation import validate_opening


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class OpeningRepairTests(unittest.TestCase):
    def opening(self, name):
        return {
            "location": "リビング", "narration": "響子はソファに座っている。キッチンから煮物の香りが漂う。",
            "character_should_reply": False,
            "entities": [{"key": "e1", "name": name, "kind": "person" if name == "五十嵐響子" else "object",
                          "perceived_by": ["player"], "description_by_view": {"player": "見える対象"}}],
        }

    def test_reported_openings_are_accepted_without_regeneration_or_unintroduced_entities(self):
        for name in ("五十嵐響子", "水が入ったコップ"):
            with self.subTest(name=name):
                invalid = self.opening(name)
                client = Mock()
                client.chat.completions.create.return_value = completion("```json\n" + json.dumps(invalid, ensure_ascii=False) + "\n```")
                result = OpenAIGameLLM(client, "test").opening({"character_name": "五十嵐響子"})
                self.assertEqual(result["narration"], invalid["narration"])
                self.assertEqual(result["entities"], [])
                self.assertEqual(client.chat.completions.create.call_count, 1)
                validate_opening(result)

    def test_duplicate_key_has_its_own_repair_reason(self):
        output = self.opening("ソファ")
        output["entities"].append(copy.deepcopy(output["entities"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate.*e1"):
            validate_opening(output)

    def test_missing_essential_narration_is_still_rejected(self):
        client = Mock()
        invalid = self.opening("水が入ったコップ")
        del invalid["narration"]
        client.chat.completions.create.return_value = completion(json.dumps(invalid))
        with self.assertRaises(GameError) as error:
            OpenAIGameLLM(client, "test").opening({"character_name": "五十嵐響子"})
        self.assertEqual(error.exception.code, "INVALID_MODEL_OUTPUT")
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_essential_failure_still_sends_specific_repair_reason(self):
        client = Mock()
        invalid = self.opening("ソファ")
        invalid["narration"] = 123
        client.chat.completions.create.side_effect = [completion(json.dumps(invalid)), completion(json.dumps(self.opening("ソファ")))]
        OpenAIGameLLM(client, "test").opening({"character_name": "五十嵐響子"})
        repair = json.loads(client.chat.completions.create.call_args.kwargs["messages"][-1]["content"])
        self.assertIn("narration", repair["validation_error"])


class CharacterReplyTests(unittest.TestCase):
    LOOPED_REPLY = (
        "（🎶）ふぅ……。😌 やっぱり、昔のゲームには特別な味がありますねっ！✨ "
        "なんか懐かしすぎて、つい鼻歌が出ちゃいました…🤭。"
        "プロデューサーさんは、特にどんな作品を思い出しますか？😊"
    )

    def payload(self):
        return {
            "messages": [{"role": "system", "content": "人物設定"}, {"role": "user", "content": "別の道を探そう"}],
            "previous_reply": "窓から風が来るね。",
        }

    def test_exact_repeat_is_retried_with_current_input_and_previous_reply(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            completion("窓から風が来るね。"), completion("それなら、廊下の先を確かめよう。"),
        ]
        result = OpenAIGameLLM(client, "test").character_reply(self.payload())
        self.assertEqual(result, "それなら、廊下の先を確かめよう。")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertTrue(any(message["content"] == "別の道を探そう" for message in messages))
        self.assertIn("窓から風が来るね。", messages[-1]["content"])

    def test_persistent_repeat_does_not_cancel_an_otherwise_valid_turn(self):
        client = Mock()
        client.chat.completions.create.return_value = completion("窓から風が来るね。")
        self.assertEqual(OpenAIGameLLM(client, "test").character_reply(self.payload()), "窓から風が来るね。")
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_different_reply_needs_no_retry(self):
        client = Mock()
        client.chat.completions.create.return_value = completion("廊下へ行ってみよう。")
        self.assertEqual(OpenAIGameLLM(client, "test").character_reply(self.payload()), "廊下へ行ってみよう。")
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_reported_loop_is_retried_despite_punctuation_or_small_word_changes(self):
        for repeated in (
            self.LOOPED_REPLY.replace("ふぅ……。😌", "ふぅ……😌"),
            self.LOOPED_REPLY.replace("特別な味", "特別な魅力"),
        ):
            with self.subTest(repeated=repeated):
                client = Mock()
                new_reply = "いい匂いでしょう？もう少し煮たら、味を見てくださいね。"
                client.chat.completions.create.side_effect = [completion(repeated), completion(new_reply)]
                payload = self.payload()
                payload["previous_reply"] = self.LOOPED_REPLY
                payload["messages"][-1]["content"] = "この煮物、いい匂いだな✨"
                self.assertEqual(OpenAIGameLLM(client, "test").character_reply(payload), new_reply)
                self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_short_replies_with_different_meanings_are_not_treated_as_repeats(self):
        client = Mock()
        client.chat.completions.create.return_value = completion("行きません。")
        payload = self.payload()
        payload["previous_reply"] = "行きます。"
        self.assertEqual(OpenAIGameLLM(client, "test").character_reply(payload), "行きません。")
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_streaming_shows_only_content_and_resets_before_repair(self):
        class Stream:
            def __init__(self, pieces):
                self.pieces = pieces
                self.closed = False
            def __iter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content="private reasoning"))])
                for piece in self.pieces:
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
            def close(self):
                self.closed = True
        first, second = Stream(["GM:", "説明"]), Stream(["窓を", "見てみよう。"])
        client = Mock()
        client.chat.completions.create.side_effect = [first, second]
        llm = OpenAIGameLLM(client, "test")
        events = []; llm.set_event_sink(events.append)
        self.assertEqual(llm.character_reply(self.payload()), "窓を見てみよう。")
        self.assertTrue(first.closed and second.closed)
        self.assertTrue(client.chat.completions.create.call_args.kwargs["stream"])
        self.assertNotIn("private reasoning", str(events))
        reset = next(i for i, event in enumerate(events) if event.get("retry"))
        self.assertEqual(events[reset]["phase"], "character")
        self.assertEqual(events[-1], {"type": "character_preview", "text": "窓を見てみよう。"})
