import unittest

from game_mode.engine import GameEngine
from game_mode.validation import validate_choices


class ChoiceContractTests(unittest.TestCase):
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
