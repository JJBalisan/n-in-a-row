"""#57: selfplay builds its agents once, not once per game.

The bar (audit N2, #57) is identical records per seed: the expected records
come from the per-game construction selfplay used before the fix. Counts and
object identities only; no wall-clock timing.
"""

import random
import unittest
from unittest import mock

from n_in_a_row import agents, arena


def _per_game_reference(spec, games, rows, cols, k, seed):
    """The pre-fix selfplay: two fresh agents per game."""
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        a = agents.load_agent(spec, random.Random(rng.getrandbits(32)))
        b = agents.load_agent(spec, random.Random(rng.getrandbits(32)))
        out.append(arena.play_game(a, b, rows, cols, k))
    return out


_SPECS = [
    ("random", 3, 3, 3),
    ("greedy", 3, 3, 3),
    ("noisy:0.25:perfect", 3, 3, 3),
    ("noisy:0.3:greedy", 3, 3, 3),
    ("noisy:0.3:negamax:2", 4, 4, 4),
    ("noisy:0.3:negamax:3", 4, 4, 3),
]


class TestSelfplayReuse(unittest.TestCase):
    def _count_loads(self, spec, games):
        real = agents.load_agent
        with mock.patch.object(agents, "load_agent", side_effect=real) as spy:
            list(arena.selfplay(spec, games, seed=1))
        return spy.call_count

    def test_construction_count_does_not_grow_with_games(self):
        for spec in ("random", "noisy:0.25:perfect", "negamax:2"):
            with self.subTest(spec=spec):
                two = self._count_loads(spec, 2)
                eight = self._count_loads(spec, 8)
                self.assertEqual(
                    eight,
                    two,
                    f"load_agent calls grow with games: {two} for 2, "
                    f"{eight} for 8",
                )

    def test_search_engine_kept_across_games(self):
        engines = []  # Keep references so identities are not reused.
        real_reset = agents.Agent.reset

        def spy(self, board):
            if isinstance(self, agents.NegamaxAgent):
                engines.append(self.engine)
            return real_reset(self, board)

        with mock.patch.object(agents.Agent, "reset", spy):
            list(arena.selfplay("noisy:0.25:perfect", 6, seed=2))
        distinct = {id(e) for e in engines}
        self.assertTrue(engines)
        self.assertLessEqual(
            len(distinct),
            2,
            f"{len(distinct)} search engines over 6 games: rebuilt per game",
        )

    def test_two_seats_stay_distinct_objects(self):
        per_game = []
        real_reset = agents.Agent.reset

        def spy(self, board):
            per_game.append(self)
            return real_reset(self, board)

        with mock.patch.object(agents.Agent, "reset", spy):
            for _ in arena.selfplay("noisy:0.25:perfect", 3, seed=4):
                pass
        self.assertEqual(len(per_game), 6)
        for g in range(3):
            x, o = per_game[2 * g], per_game[2 * g + 1]
            self.assertIsNot(x, o, f"game {g}: one object in both seats")

    def test_records_identical_to_per_game_construction(self):
        for spec, rows, cols, k in _SPECS:
            for seed in (7, 11):
                with self.subTest(spec=spec, seed=seed):
                    want = _per_game_reference(spec, 12, rows, cols, k, seed)
                    got = list(
                        arena.selfplay(spec, 12, rows, cols, k, seed=seed)
                    )
                    self.assertEqual(
                        [r.to_json() for r in got],
                        [r.to_json() for r in want],
                    )


if __name__ == "__main__":
    unittest.main()
