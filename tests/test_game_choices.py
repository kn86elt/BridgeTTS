import unittest

from game_mode.engine import GameEngine
from game_mode.validation import validate_choices, validate_resolution
from game_mode.output import coerce_model_output, normalize_opening, normalize_resolution, prepare_choices


class CoercionTests(unittest.TestCase):
    def resolution(self):
        return {"kind": "resolve", "intent": "action", "narration": "キッチンに煮物と秘密の箱が見える。",
                "character_event": "煮物の匂いがする。", "action_succeeded": True, "character_should_reply": True}

    def test_resolution_variants_are_mapped_onto_the_contract(self):
        out = coerce_model_output("resolve", {
            "kind": "Resolve", "intent": "Action", "reasoning": "ignored", "narration": None,
            "action_succeeded": "true", "character_should_reply": "no", "target_key": "null", "hook": "",
            "important_facts": [{"text": "x", "known_by": "player"}, {"text": "bad"}],
        })
        self.assertEqual(out["kind"], "resolve")
        self.assertEqual(out["intent"], "action")
        self.assertNotIn("reasoning", out)
        self.assertEqual(out["narration"], "")
        self.assertTrue(out["action_succeeded"])
        self.assertFalse(out["character_should_reply"])
        self.assertIsNone(out["target_key"])
        self.assertIsNone(out["hook"])
        self.assertEqual(out["important_facts"], [{"text": "x", "known_by": ["player"]}])
        validate_resolution(out)

    def test_clarify_is_inferred_and_singular_target_key_is_accepted(self):
        out = coerce_model_output("resolve", {"question": "どれ？", "command": "LOOK", "target_key": "window"})
        self.assertEqual(out, {"kind": "clarify", "question": "どれ？", "command": "look", "target_keys": ["window"]})

    def test_choice_aliases_and_bare_lists_are_accepted(self):
        out = coerce_model_output("choices", [
            {"type": "say", "label": "「ねえ」"}, "「行こう」", "扉を開ける",
        ])
        self.assertEqual(out["choices"][0], {"kind": "speech", "text": "ねえ"})
        self.assertEqual(out["choices"][1], {"kind": "speech", "text": "行こう"})
        self.assertEqual(out["choices"][2], {"kind": "action", "text": "扉を開ける"})

    def test_opening_defaults_reply_flag_from_character_event(self):
        out = coerce_model_output("opening", {"location": "部屋", "narration": "静かだ。", "character_event": "物音", "extra": 1})
        self.assertTrue(out["character_should_reply"])
        self.assertNotIn("extra", out)

    def test_direct_normalize_still_rejects_invalid_flags(self):
        # Coercion belongs to llm.py; normalize_resolution stays strict.
        with self.assertRaises(ValueError):
            normalize_resolution({**self.resolution(), "action_succeeded": "yes"}, {})


class ChoiceContractTests(unittest.TestCase):
    def resolution(self):
        return {"kind": "resolve", "intent": "action", "narration": "キッチンに煮物と秘密の箱が見える。",
                "character_event": "煮物の匂いがする。", "action_succeeded": True, "character_should_reply": True}

    def test_optional_resolution_entities_do_not_infer_shared_perception(self):
        output = self.resolution()
        output["entity_additions"] = [
            {"key": "kitchen", "name": "キッチン", "kind": "place", "perceived_by": ["player", "character"],
             "description_by_view": {"player": "", "character": ""}},
            {"key": "box", "name": "秘密の箱", "kind": "object", "perceived_by": ["player", "character"],
             "description": "共通の説明から視点を推測してはいけない"},
            {"key": "stew", "name": "煮物", "kind": "object", "perceived_by": ["player", "character"],
             "description_by_view": {"player": "鍋に入っている", "character": "匂いがする"}},
        ]
        result = normalize_resolution(output, {})
        self.assertEqual([entity["key"] for entity in result["entity_additions"]], ["stew"])
        self.assertEqual(result["narration"], output["narration"])

    def test_unannounced_or_existing_resolution_entities_are_discarded(self):
        output = self.resolution()
        output["entity_additions"] = [
            {"key": "box", "name": "秘密の箱", "kind": "object", "perceived_by": ["character"],
             "description_by_view": {"character": "キャラには知らされていない"}},
            {"key": "stew", "name": "煮物", "kind": "object", "perceived_by": ["player"],
             "description_by_view": {"player": "見える"}},
        ]
        result = normalize_resolution(output, {"entities": [{"key": "stew"}]})
        self.assertEqual(result["entity_additions"], [])

    def test_resolution_core_still_requires_valid_outcome_flags(self):
        output = self.resolution()
        output["action_succeeded"] = "yes"
        with self.assertRaises(ValueError):
            normalize_resolution(output, {})

    def choices(self, text, extra_step=False):
        steps = [{"kind": "speech", "text": text}]
        if extra_step:
            steps.append({"kind": "action", "text": "扉を開ける"})
        return {"choices": [{"steps": steps} for _ in range(3)]}

    def test_short_choices_keep_display_and_execution_identical(self):
        output = self.choices("あ" * 40)
        validate_choices(output)
        choices = GameEngine._new_choices(output, {"entities": []})
        self.assertEqual(choices[0].label, "「" + "あ" * 40 + "」")
        self.assertEqual(choices[0].selection["steps"][0]["text"], "あ" * 40)

    def test_long_or_combined_choices_are_rejected_before_leaving_llm_validation(self):
        for output in (self.choices("あ" * 41), self.choices("話しかける", extra_step=True)):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate_choices(output)

    def test_optional_opening_metadata_does_not_expand_perception(self):
        output = normalize_opening({
            "narration": "窓が見える。", "extra": "ignored",
            "entities": [{"name": "窓", "kind": "object", "perceived_by": ["player"],
                          "description_by_view": {"player": "開いている", "character": "知らない説明"}, "extra": 1},
                         {"name": "窓", "kind": "object", "perceived_by": ["player", "character"]}],
            "important_facts": [{"text": "秘密", "known_by": ["player"]}, {"text": "視点未指定"}],
        }, "ミオ")
        self.assertEqual(output["entities"][0]["perceived_by"], ["player"])
        self.assertEqual(len(output["entities"]), 1)
        self.assertEqual(output["important_facts"], [{"text": "秘密", "known_by": ["player"]}])

    def test_partial_compact_choices_keep_valid_text_and_drop_unsafe_candidates(self):
        scene = {"entities": [{"key": "window", "id": "window-id", "perceived_by": ["player"]},
                              {"key": "secret", "id": "secret-id", "perceived_by": ["character"]}]}
        output, supplemented = prepare_choices({"choices": [
            {"kind": "action", "text": "窓を開ける", "target_key": "window", "explanation": "ignored"},
            {"kind": "action", "text": "秘密を探る", "target_key": "secret"},
            {"kind": "action", "text": "あ" * 41},
            {"steps": [{"kind": "action", "text": "歩く"}, {"kind": "speech", "text": "待って"}]},
        ]}, scene)
        self.assertTrue(supplemented)
        choices = GameEngine._new_choices(output, scene)
        self.assertEqual(choices[0].label, "窓を開ける")
        self.assertEqual(choices[0].selection["steps"][0]["target_id"], "window-id")
        self.assertEqual(len(choices), 3)
        self.assertTrue(all(choice.selection["steps"][0]["kind"] == "speech" for choice in choices[1:]))

    def test_content_fallback_is_speech_only_and_duplicates_are_removed(self):
        output, supplemented = prepare_choices({"choices": [
            {"kind": "action", "text": "歩く"},
            {"kind": "speech", "text": "教えて"}, {"kind": "speech", "text": "教えて"},
        ]}, {}, content_only=True)
        self.assertTrue(supplemented)
        texts = [choice["steps"][0]["text"] for choice in output["choices"]]
        self.assertEqual(len(set(texts)), 3)
        self.assertTrue(all(choice["steps"][0]["kind"] == "speech" for choice in output["choices"]))
