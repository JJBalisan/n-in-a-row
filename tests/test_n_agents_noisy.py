"""Checks for #33 and #45: NoisyAgent forwarding and noisy: spec folding.

Stdlib only:  python3 -m unittest discover -s tests
"""

import os
import random
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents, arena, game


class Recorder(agents.Agent):
    """Counts the hooks it is sent; plays the first legal move."""

    name = "recorder"

    def __init__(self):
        self.resets = 0
        self.observed = []

    def reset(self, board):
        self.resets += 1

    def observe(self, board, move):
        self.observed.append(move)

    def select(self, board):
        return board.legal_moves()[0]


class FirstCell(agents.NegamaxAgent):
    """A NegamaxAgent subclass whose select ignores epsilon."""

    def select(self, board):
        return board.legal_moves()[0]


# Module-level instances a ``noisy:P:module:attribute`` spec can reach.
SHARED = agents.NegamaxAgent(depth=2)
SHARED_SUB = FirstCell(depth=2)


def _spec(attribute):
    return f"noisy:0.5:{__name__}:{attribute}"


class TestNoisyAgentForwardsHooks(unittest.TestCase):
    """#33: the wrapped agent must be told about the game."""

    def test_base_is_reset_once_per_game(self):
        rec = Recorder()
        noisy = agents.NoisyAgent(rec, 0.0, random.Random(0))
        arena.play_game(noisy, agents.RandomAgent(random.Random(1)))
        self.assertEqual(rec.resets, 1)

    def test_base_observes_every_move(self):
        rec = Recorder()
        noisy = agents.NoisyAgent(rec, 0.0, random.Random(0))
        record = arena.play_game(noisy, agents.RandomAgent(random.Random(1)))
        self.assertEqual(rec.observed, list(record.moves))

    def test_base_observes_random_moves_too(self):
        rec = Recorder()
        noisy = agents.NoisyAgent(rec, 1.0, random.Random(3))
        record = arena.play_game(noisy, agents.RandomAgent(random.Random(4)))
        self.assertEqual(rec.resets, 1)
        self.assertEqual(rec.observed, list(record.moves))

    def test_direct_hook_calls_forward(self):
        rec = Recorder()
        agent = agents.NoisyAgent(rec, 0.0, random.Random(0))
        agent.reset(game.Board())
        agent.observe(game.Board.from_moves([4]), 4)
        self.assertEqual((rec.resets, rec.observed), (1, [4]))


class Explains(Recorder):
    """A recorder whose rationale is a fixed sentinel."""

    def explain(self, board):
        return "SENTINEL"


class TestNoisyAgentExplain(unittest.TestCase):
    """#33: explain after a random move must not report the base's search."""

    def test_random_move_is_not_explained_by_the_base(self):
        agent = agents.NoisyAgent(Explains(), 1.0, random.Random(0))
        board = game.Board()
        agent.select(board)
        self.assertNotEqual(agent.explain(board), "SENTINEL")
        # The guard against a fix that stops delegating altogether.
        agent = agents.NoisyAgent(Explains(), 0.0, random.Random(0))
        agent.select(board)
        self.assertEqual(agent.explain(board), "SENTINEL")


class TestLoadNoisyDoesNotMutate(unittest.TestCase):
    """#45: noisy: must not change the agent a spec resolves to."""

    def test_shared_instance_is_left_alone(self):
        SHARED.epsilon = 0.0
        SHARED.name = "negamax:2"
        agents.load_agent(_spec("SHARED"), random.Random(0))
        self.assertEqual(SHARED.epsilon, 0.0)
        self.assertEqual(SHARED.name, "negamax:2")

    def test_subclass_overriding_select_still_plays_noise(self):
        agent = agents.load_agent(
            f"noisy:1:{__name__}:SHARED_SUB", random.Random(0)
        )
        board = game.Board()
        picks = {agent.select(board) for _ in range(50)}
        self.assertGreater(len(picks), 1, "epsilon 1 played one cell only")

    def test_nested_noise_compounds(self):
        # X wins at cell 2; perfect always plays it. Five legal moves, so
        # a random move misses the win 4/5 of the time. Compounded noise
        # 1 - 0.9 * 0.6 = 0.46 misses ~0.37; the outer 0.1 alone ~0.08.
        agent = agents.load_agent(
            "noisy:0.1:noisy:0.4:perfect", random.Random(0)
        )
        board = game.Board.from_moves([0, 3, 1, 4])
        misses = sum(agent.select(board) != 2 for _ in range(1000))
        self.assertGreater(misses, 250)
        self.assertLess(misses, 500)


if __name__ == "__main__":
    unittest.main()
