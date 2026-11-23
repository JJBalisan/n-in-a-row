"""#57 review: selfplay reuses built-in agents only.

Reseeding one object per game stands in for a fresh pair only when the
agent shares the one Random it was given and keeps nothing else between
games. A ``module:attribute`` agent promises neither: it may keep state
that only a new object clears, or take its seed from the Random it is
built with. For those, selfplay must give the records the per-game
construction gave before #57, and the same records for the same seed.
"""

import random
import sys
import types
import unittest

from n_in_a_row import agents, arena

_MOD = "_n_cli_arena_selfplay_module_fixture"


class Tired(agents.Agent):
    """Plays the last legal move from its fourth move on; has no reset."""

    name = "tired"

    def __init__(self):
        self.moves = 0

    def select(self, board):
        self.moves += 1
        legal = board.legal_moves()
        return legal[-1] if self.moves > 3 else legal[0]


class SelfSeeded(agents.Agent):
    """Seeds its own generator from one draw of the Random it is given."""

    name = "self-seeded"

    def __init__(self, rng=None):
        source = rng or random.Random(0)
        self.rng = random.Random(source.getrandbits(32))

    def select(self, board):
        return self.rng.choice(board.legal_moves())


def self_seeded(rng=None):
    return SelfSeeded(rng)


def _install():
    mod = types.ModuleType(_MOD)
    mod.Tired = Tired
    mod.SelfSeeded = SelfSeeded
    mod.self_seeded = self_seeded
    sys.modules[_MOD] = mod


def _per_game_reference(spec, games, seed):
    """The pre-#57 selfplay: two fresh agents per game."""
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        a = agents.load_agent(spec, random.Random(rng.getrandbits(32)))
        b = agents.load_agent(spec, random.Random(rng.getrandbits(32)))
        out.append(arena.play_game(a, b))
    return [r.to_json() for r in out]


def _selfplay(spec, games, seed):
    return [r.to_json() for r in arena.selfplay(spec, games, seed=seed)]


class TestModuleAgentsBuiltPerGame(unittest.TestCase):
    def setUp(self):
        _install()
        self.addCleanup(sys.modules.pop, _MOD, None)

    def test_state_kept_by_a_class_does_not_leak_between_games(self):
        spec = f"{_MOD}:Tired"
        want = _per_game_reference(spec, 4, 1)
        self.assertEqual(_selfplay(spec, 4, 1), want)

    def test_noisy_wrapped_module_agent_matches_per_game(self):
        for spec in (
            f"noisy:0.3:{_MOD}:Tired",
            f"noisy:0:noisy:0:{_MOD}:Tired",
        ):
            with self.subTest(spec=spec):
                want = _per_game_reference(spec, 6, 2)
                self.assertEqual(_selfplay(spec, 6, 2), want)

    def test_seeded_run_repeats_for_a_self_seeding_agent(self):
        for attr in ("SelfSeeded", "self_seeded"):
            spec = f"{_MOD}:{attr}"
            with self.subTest(spec=spec):
                first = _selfplay(spec, 8, 3)
                self.assertEqual(_selfplay(spec, 8, 3), first)
                self.assertEqual(first, _per_game_reference(spec, 8, 3))


if __name__ == "__main__":
    unittest.main()
