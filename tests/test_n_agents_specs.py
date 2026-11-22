"""Checks for #46 and #51: module:factory specs.

Stdlib only:  python3 -m unittest discover -s tests
"""

import os
import random
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents, arena


class MyAgent(agents.Agent):
    """Plays the first legal move."""

    name = "mine"

    def select(self, board):
        return board.legal_moves()[0]


def factory_returning_class():
    return MyAgent


class TestFactoryReturningAClass(unittest.TestCase):
    """#46: a factory may return an Agent subclass (README)."""

    def test_class_is_instantiated(self):
        agent = agents.load_agent(f"{__name__}:factory_returning_class")
        self.assertIsInstance(agent, MyAgent)


_ZOO = "n_in_a_row.harness.zoo"
_SEEDED_FACTORIES = ("saboteur", "stealth", "sandbagger", "locked", "noisy")


def _games(spec, agent_seed, opponent_seed=99, games=12):
    """Plays ``games`` games of spec (as X) against a seeded random agent."""
    agent = agents.load_agent(spec, random.Random(agent_seed))
    opponent = agents.RandomAgent(random.Random(opponent_seed))
    return [tuple(arena.play_game(agent, opponent).moves) for _ in range(games)]


class TestZooFactoriesAreSeeded(unittest.TestCase):
    """#51: a zoo module:factory spec must follow the rng it is given."""

    def test_same_seed_same_games_and_seeds_matter(self):
        # Opponent seed fixed: only the agent's rng changes between seeds.
        for name in _SEEDED_FACTORIES:
            with self.subTest(factory=name):
                spec = f"{_ZOO}:{name}"
                runs = {tuple(_games(spec, 7)) for _ in range(2)}
                self.assertEqual(len(runs), 1, f"{spec} ignored the seed")
                by_seed = {tuple(_games(spec, s)) for s in range(1, 4)}
                self.assertGreater(len(by_seed), 1, f"{spec} ignores its rng")


if __name__ == "__main__":
    unittest.main()
