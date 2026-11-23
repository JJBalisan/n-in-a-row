"""#49 (the from_json half): a GameRecord read back from its JSON line.

The other half of #49, what result_for gives when winner is None, waits on
JJ (found/49-decision.md in the sweep lane folder) and is not checked here.
"""

import contextlib
import io
import os
import random
import tempfile
import unittest

from n_in_a_row import agents, arena, cli


class TestFromJson(unittest.TestCase):
    def test_has_from_json(self):
        self.assertTrue(
            hasattr(arena.GameRecord, "from_json"),
            "GameRecord has to_json but no from_json",
        )

    def test_round_trip_played_records(self):
        self.assertTrue(hasattr(arena.GameRecord, "from_json"))
        rng = random.Random(3)
        for rows, cols, k in [(3, 3, 3), (3, 4, 3), (4, 4, 3)]:
            for _ in range(5):
                rec = arena.play_game(
                    agents.RandomAgent(rng),
                    agents.RandomAgent(rng),
                    rows,
                    cols,
                    k,
                )
                back = arena.GameRecord.from_json(rec.to_json())
                self.assertIsInstance(back, arena.GameRecord)
                self.assertEqual(back, rec)

    def test_reads_match_out_file(self):
        self.assertTrue(hasattr(arena.GameRecord, "from_json"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "games.jsonl")
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(
                    [
                        "match",
                        "--a",
                        "random",
                        "--b",
                        "greedy",
                        "--games",
                        "4",
                        "--seed",
                        "5",
                        "--out",
                        path,
                        "--quiet",
                    ]
                )
            with open(path) as fh:
                lines = [ln for ln in fh.read().splitlines() if ln]
        self.assertEqual(len(lines), 4)
        for line in lines:
            rec = arena.GameRecord.from_json(line)
            self.assertEqual(rec.to_json(), line)
            self.assertIn(rec.winner, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
