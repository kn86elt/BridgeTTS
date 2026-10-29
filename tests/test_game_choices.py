import unittest

from game_mode.engine import GameEngine
from game_mode.validation import validate_choices
from game_mode.output import normalize_opening, normalize_resolution, prepare_choices


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
