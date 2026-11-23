"""#57 review: a depth-capped search starts every selfplay game cold.

A search that stops short of the end of the game reads deeper table
entries left by earlier games and then plays other moves, so a table
kept across games changes the records for a seed. Only a search that
runs to the end of the game on this board may keep its table; the
records must be those of per-game construction either way.
"""

import random
import unittest
from unittest import mock

from n_in_a_row import agents, arena


def _per_game_reference(spec, games, rows, cols, k, seed):
    """The pre-#57 selfplay: two fresh agents per game."""
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        a = agents.load_agent(spec, random.Random(rng.getrandbits(32)))
        b = agents.load_agent(spec, random.Random(rng.getrandbits(32)))
        out.append(arena.play_game(a, b, rows, cols, k))
    return [r.to_json() for r in out]


def _selfplay(spec, games, rows, cols, k, seed):
    recs = arena.selfplay(spec, games, rows, cols, k, seed=seed)
    return [r.to_json() for r in recs]


def _engines(spec, games, rows, cols, k):
    """The search engine of every negamax agent at each reset."""
    seen = []
    real_reset = agents.Agent.reset

    def spy(agent, board):
        if isinstance(agent, agents.NegamaxAgent):
            seen.append(agent.engine)
        return real_reset(agent, board)

    with mock.patch.object(agents.Agent, "reset", spy):
        list(arena.selfplay(spec, games, rows, cols, k, seed=1))
    return seen


class TestDepthCappedRecords(unittest.TestCase):
    def test_records_match_per_game_construction(self):
        for spec, rows, cols, k, seed in (
            ("noisy:0.2:negamax:4", 4, 4, 4, 1),
            ("noisy:0.3:negamax:5", 3, 3, 3, 21),
            ("noisy:0.2:noisy:0.1:negamax:4", 4, 4, 4, 1),
        ):
            with self.subTest(spec=spec, seed=seed):
                want = _per_game_reference(spec, 8, rows, cols, k, seed)
                got = _selfplay(spec, 8, rows, cols, k, seed)
                self.assertEqual(got, want)

    def test_capped_search_keeps_its_settings(self):
        engines = _engines("noisy:0.2:negamax:4", 3, 4, 4, 4)
        self.assertTrue(engines)
        for engine in engines:
            self.assertEqual(engine.max_depth, 4)
            self.assertIsNone(engine.time_limit)

    def test_search_to_the_end_of_the_game_keeps_its_table(self):
        for spec, rows, cols, k in (
            ("noisy:0.25:perfect", 3, 3, 3),
            ("noisy:0.3:negamax:9", 3, 3, 3),
            ("noisy:0.3:perfect", 3, 4, 3),
        ):
            with self.subTest(spec=spec):
                engines = _engines(spec, 4, rows, cols, k)
                self.assertLessEqual(len({id(e) for e in engines}), 2)


if __name__ == "__main__":
    unittest.main()
