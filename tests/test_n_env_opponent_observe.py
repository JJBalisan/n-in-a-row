"""Checks for #42: TicTacToeEnv tells its opponent about every move."""

import os
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))
sys.path.insert(0, _TESTS)

from n_in_a_row import agents, game

try:
    from n_in_a_row import env
except ModuleNotFoundError as err:
    if err.name != "numpy":
        raise
    env = None  # setUp skips every test, or fails it when CI is set


def _require_numpy(test):
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        if os.environ.get("CI"):
            test.fail(
                "numpy is not installed but CI is set; "
                "install the test extra: pip install -e '.[test]'"
            )
        test.skipTest("numpy not installed (set CI=1 to make this a failure)")


class Recorder(agents.Agent):
    """Plays the highest empty cell and records every hook call."""

    name = "recorder"

    def __init__(self):
        self.resets = 0
        self.seen = []  # (ply, move, last move on the board) at call time

    def reset(self, board):
        self.resets += 1
        self.seen = []

    def observe(self, board, move):
        last = board.moves_played()[-1] if board.ply else None
        self.seen.append((board.ply, move, last))

    def select(self, board):
        return board.legal_moves()[-1]


def _expected(moves):
    return [(i + 1, m, m) for i, m in enumerate(moves)]


class TicTacToeEnvObservesOpponentTest(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)

    def test_opponent_observes_every_move_by_both_sides(self):
        opp = Recorder()
        e = env.TicTacToeEnv(3, opponent=opp, agent_plays=game.X)
        e.reset()
        done = False
        while not done:
            _, _, done, _ = e.step(e.legal_actions()[0])
        # X 0, O 8, X 1, O 7, X 2: the trained agent's move ends the game.
        self.assertEqual(e.board.moves_played(), [0, 8, 1, 7, 2])
        self.assertEqual(opp.resets, 1)
        self.assertEqual(opp.seen, _expected(e.board.moves_played()))

    def test_opponent_holding_x_observes_its_opening_move(self):
        opp = Recorder()
        e = env.TicTacToeEnv(3, opponent=opp, agent_plays=game.O)
        e.reset()
        self.assertEqual(e.board.moves_played(), [8])
        self.assertEqual(opp.seen, [(1, 8, 8)])
        done = False
        while not done:
            _, _, done, _ = e.step(e.legal_actions()[0])
        self.assertEqual(opp.seen, _expected(e.board.moves_played()))

    def test_observe_counts_restart_with_each_game(self):
        opp = Recorder()
        e = env.TicTacToeEnv(3, opponent=opp, agent_plays=game.X)
        for swap in (False, True):
            e.reset(swap_sides=swap)
            done = False
            while not done:
                _, _, done, _ = e.step(e.legal_actions()[0])
            self.assertEqual(opp.seen, _expected(e.board.moves_played()))
        self.assertEqual(opp.resets, 2)


if __name__ == "__main__":
    unittest.main()
