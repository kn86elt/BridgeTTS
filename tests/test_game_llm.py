import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from game_mode.llm import OpenAIGameLLM, _decode_json, strip_reasoning
import game_mode.llm as game_llm
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
        invalid["narration"] = "x" * 301
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

    def test_reasoning_block_is_removed_from_character_reply_and_preview(self):
        class Stream:
            def __init__(self, pieces): self.pieces = pieces
            def __iter__(self):
                for piece in self.pieces:
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
            def close(self): pass
        client = Mock()
        client.chat.completions.create.return_value = Stream(["<think>", "内心", "</think>", "廊下へ", "行こう。"])
        llm = OpenAIGameLLM(client, "test")
        events = []; llm.set_event_sink(events.append)
        self.assertEqual(llm.character_reply(self.payload()), "廊下へ行こう。")
        self.assertFalse(any("内心" in str(event) for event in events))
        self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], game_llm.MAX_TOKENS["character"])


class DecodeTests(unittest.TestCase):
    def test_reasoning_block_and_prose_around_json_are_ignored(self):
        text = "<think>長い思考</think>はい、以下です。\n```json\n{\"kind\":\"clarify\",\"question\":\"どれ？\",\"command\":\"look\"}\n```\n以上です。"
        self.assertEqual(_decode_json(text)["kind"], "clarify")

    def test_unterminated_reasoning_block_yields_nothing(self):
        self.assertEqual(strip_reasoning("<think>まだ考え中"), "")

    def test_first_balanced_object_in_prose_is_used(self):
        self.assertEqual(_decode_json('結果: {"a": {"b": 1}} 補足 {"c": 2}'), {"a": {"b": 1}})

    def test_bare_array_is_returned_for_choices_coercion(self):
        self.assertIsInstance(_decode_json('[{"kind":"speech","text":"ねえ"}]'), list)

    def test_prose_prefixed_bare_array_is_returned_for_choices_coercion(self):
        self.assertEqual(
            _decode_json('候補: [{"kind":"speech","text":"ねえ"}]'),
            [{"kind": "speech", "text": "ねえ"}],
        )

    def test_prose_without_json_raises(self):
        with self.assertRaises(ValueError):
            _decode_json("JSONは出せません。")


class ResponseFormatTests(unittest.TestCase):
    VALID = '{"text":"要約","source_event_ids":[]}'

    def setUp(self):
        with game_llm._backend_lock:
            game_llm._format_support.clear()
            game_llm._reasoning_effort_support.clear()
            game_llm._reasoning_allowance.clear()

    def rejection(self, status=400):
        exc = Exception("response_format not supported")
        exc.status_code = status
        return exc

    def test_json_schema_is_tried_first_then_downgraded_and_remembered(self):
        client = Mock()
        client.chat.completions.create.side_effect = [self.rejection(), completion(self.VALID), completion(self.VALID)]
        llm = OpenAIGameLLM(client, "test")
        llm.summarize({"view": "player"})
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(calls[1].kwargs["response_format"]["type"], "json_object")
        # Same client/model: the remembered mode is used directly.
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        self.assertEqual(calls[2].kwargs["response_format"]["type"], "json_object")

    def test_plain_text_fallback_omits_response_format(self):
        client = Mock()
        client.chat.completions.create.side_effect = [self.rejection(), self.rejection(), completion(self.VALID)]
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        self.assertNotIn("response_format", client.chat.completions.create.call_args.kwargs)

    def test_timeout_is_not_treated_as_format_rejection(self):
        client = Mock()
        client.chat.completions.create.side_effect = TimeoutError()
        with self.assertRaises(GameError) as error:
            OpenAIGameLLM(client, "test").summarize({"view": "player"})
        self.assertEqual(error.exception.code, "LLM_TIMEOUT")
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_json_roles_send_caps_and_low_temperature(self):
        client = Mock()
        client.chat.completions.create.return_value = completion(self.VALID)
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], game_llm.JSON_TEMPERATURE)
        self.assertEqual(kwargs["max_tokens"], game_llm.MAX_TOKENS["summary"])

    def test_repair_appends_to_the_same_conversation(self):
        client = Mock()
        client.chat.completions.create.side_effect = [completion("壊れた出力"), completion(self.VALID)]
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], "壊れた出力")
        self.assertIn("validation_error", json.loads(messages[3]["content"]))

    def test_reasoning_effort_none_is_sent_with_every_call(self):
        client = Mock()
        client.chat.completions.create.return_value = completion(self.VALID)
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        self.assertEqual(client.chat.completions.create.call_args.kwargs["extra_body"], game_llm.REASONING_OFF_BODY)

    def test_reasoning_off_body_covers_both_template_families(self):
        self.assertEqual(game_llm.REASONING_OFF_BODY["reasoning_effort"], "none")
        self.assertIs(game_llm.REASONING_OFF_BODY["chat_template_kwargs"]["enable_thinking"], False)

    def test_rejected_reasoning_effort_is_dropped_and_remembered(self):
        client = Mock()
        rejected = Exception("Unsupported parameter: 'reasoning_effort'")
        rejected.status_code = 400
        client.chat.completions.create.side_effect = [rejected, completion(self.VALID), completion(self.VALID)]
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        calls = client.chat.completions.create.call_args_list
        self.assertIn("extra_body", calls[0].kwargs)
        self.assertNotIn("extra_body", calls[1].kwargs)
        self.assertEqual(calls[1].kwargs["response_format"]["type"], "json_schema")
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        self.assertNotIn("extra_body", calls[2].kwargs)

    def test_format_rejection_does_not_disable_reasoning_body(self):
        client = Mock()
        rejected = Exception("Unsupported parameter: 'response_format'")
        rejected.status_code = 400
        client.chat.completions.create.side_effect = [rejected, completion(self.VALID), completion(self.VALID)]
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        calls = client.chat.completions.create.call_args_list
        self.assertIn("extra_body", calls[0].kwargs)
        self.assertIn("extra_body", calls[1].kwargs)
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        self.assertIn("extra_body", calls[2].kwargs)

    def test_unrelated_client_error_does_not_disable_reasoning_body(self):
        client = Mock()
        unauthorized = Exception("Invalid API key")
        unauthorized.status_code = 401
        client.chat.completions.create.side_effect = [self.rejection(), self.rejection(), unauthorized, completion(self.VALID)]
        llm = OpenAIGameLLM(client, "test")
        with self.assertRaises(GameError):
            llm.summarize({"view": "player"})
        calls = client.chat.completions.create.call_args_list
        self.assertIn("extra_body", calls[2].kwargs)
        self.assertNotIn(llm._format_key(), game_llm._reasoning_effort_support)
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        self.assertIn("extra_body", calls[3].kwargs)

    def test_length_cutoff_with_empty_answer_retries_with_larger_budget(self):
        truncated = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=""), finish_reason="length",
        )])
        client = Mock()
        client.chat.completions.create.side_effect = [truncated, completion(self.VALID)]
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1].kwargs["max_tokens"], calls[0].kwargs["max_tokens"])
        self.assertEqual([m["role"] for m in calls[1].kwargs["messages"]], ["system", "user"])

    def test_reasoning_allowance_is_capped_at_6000(self):
        client = Mock()
        client.chat.completions.create.return_value = completion(self.VALID)
        llm = OpenAIGameLLM(client, "test")
        for _ in range(8):
            llm._grow_reasoning_allowance()
        llm.summarize({"view": "player"})
        self.assertEqual(game_llm._reasoning_allowance[llm._format_key()], game_llm.REASONING_ALLOWANCE_MAX)
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["max_tokens"],
            game_llm.MAX_TOKENS["summary"] + game_llm.REASONING_ALLOWANCE_MAX,
        )

    def test_character_reply_also_disables_reasoning(self):
        client = Mock()
        client.chat.completions.create.return_value = completion("廊下へ行ってみよう。")
        OpenAIGameLLM(client, "test").character_reply({
            "messages": [{"role": "system", "content": "人物設定"}, {"role": "user", "content": "行こう"}],
            "previous_reply": "",
        })
        self.assertEqual(client.chat.completions.create.call_args.kwargs["extra_body"], game_llm.REASONING_OFF_BODY)

    def test_unknown_parameter_rejection_drops_reasoning_body_before_format(self):
        client = Mock()
        rejected = Exception("Unrecognized request argument supplied: chat_template_kwargs")
        rejected.status_code = 400
        client.chat.completions.create.side_effect = [rejected, completion(self.VALID)]
        OpenAIGameLLM(client, "test").summarize({"view": "player"})
        calls = client.chat.completions.create.call_args_list
        self.assertNotIn("extra_body", calls[1].kwargs)
        self.assertEqual(calls[1].kwargs["response_format"]["type"], "json_schema")
