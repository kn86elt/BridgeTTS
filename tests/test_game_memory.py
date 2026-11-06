from copy import deepcopy
import unittest

from game_mode.memory import add_turn, compact_if_needed, new_memory
from game_mode.models import GameError


class GameMemoryTests(unittest.TestCase):
    def memory(self, count=13, text="出来事"):
        memory = new_memory({})
        for turn in range(1, count + 1):
            add_turn(memory, turn, [{
                "id": f"event-{turn}", "kind": "environment",
                "text_by_view": {"player": text, "character": text},
            }])
        return memory

    def test_japanese_limit_counts_characters_not_bytes(self):
        memory = self.memory(count=6, text="あ" * 1500)
        original = deepcopy(memory)
        compact_if_needed(memory, 6, lambda *_: self.fail("No old turns to summarize"))
        self.assertEqual(memory, original)

    def test_summary_failure_preserves_both_views_and_retries(self):
        for failure in (GameError("INVALID_MODEL_OUTPUT", 502), {"text": "bad", "source_event_ids": ["unknown"]}):
            with self.subTest(failure=str(failure)):
                memory = self.memory()
                original = deepcopy(memory)

                def summarize(view, payload):
                    if view == "character":
                        if isinstance(failure, Exception):
                            raise failure
                        return failure
                    return {"text": "summary", "source_event_ids": ["event-1"]}

                compact_if_needed(memory, 13, summarize)
                self.assertEqual(memory, original)
                compact_if_needed(memory, 13, lambda *_: {"text": "summary", "source_event_ids": ["event-1"]})
                self.assertEqual(len(memory["recent_turns"]), 6)
                self.assertEqual(memory["event_ledger"], original["event_ledger"])

    def test_failed_summary_over_real_limit_remains_atomic(self):
        memory = self.memory(text="x" * 2000)
        original = deepcopy(memory)

        def fail(*_):
            raise GameError("INVALID_MODEL_OUTPUT", 502)

        with self.assertRaises(GameError) as error:
            compact_if_needed(memory, 13, fail)
        self.assertEqual(error.exception.code, "MEMORY_COMPACTION_FAILED")
        self.assertEqual(memory, original)
