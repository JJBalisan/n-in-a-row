"""Selector's adversarial checks for #36 and #42 (sweep 22 Nov)."""

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
    """Plays a fixed preference order and records every observe call."""

    name = "recorder"

    def __init__(self, order=None):
        self.order = order
        self.resets = 0
        self.seen = []

    def reset(self, board):
        self.resets += 1
        self.seen = []

    def observe(self, board, move):
        self.seen.append((board.ply, move))

    def select(self, board):
        legal = board.legal_moves()
        if self.order:
            for m in self.order:
                if m in legal:
                    return m
        return legal[-1]


def _expected(board):
    return [(i + 1, m) for i, m in enumerate(board.moves_played())]


class ObserveSelectTest(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)

    def test_illegal_move_under_lose_is_not_observed(self):
        opp = Recorder()
        e = env.TicTacToeEnv(
            3, opponent=opp, agent_plays=game.X, illegal_move="lose"
        )
        e.reset()
        e.step(0)  # X 0, O 8
        self.assertEqual(opp.seen, [(1, 0), (2, 8)])
        _, r, done, info = e.step(0)  # occupied
        self.assertTrue(done)
        self.assertEqual(r, -1.0)
        self.assertTrue(info["illegal"])
        self.assertEqual(opp.seen, [(1, 0), (2, 8)])

    def test_illegal_move_under_raise_is_not_observed(self):
        opp = Recorder()
        e = env.TicTacToeEnv(3, opponent=opp, agent_plays=game.X)
        e.reset()
        e.step(0)
        with self.assertRaises(game.InvalidMove):
            e.step(8)
        self.assertEqual(opp.seen, [(1, 0), (2, 8)])

    def test_opponent_winning_move_is_observed(self):
        # Opponent O takes 6, 7, 8; the agent X plays 0, 1, 3 and loses.
        opp = Recorder(order=[6, 7, 8])
        e = env.TicTacToeEnv(3, opponent=opp, agent_plays=game.X)
        e.reset()
        done = False
        for a in (0, 1, 3):
            _, r, done, _ = e.step(a)
        self.assertTrue(done)
        self.assertEqual(e.board.winner, game.O)
        self.assertEqual(r, -1.0)
        self.assertEqual(e.board.moves_played(), [0, 6, 1, 7, 3, 8])
        self.assertEqual(opp.seen, _expected(e.board))

    def test_drawn_game_every_move_observed(self):
        # X: 0 2 3 7 8 ; O: 1 4 5 6 -> draw.
        opp = Recorder(order=[4, 1, 5, 6])
        e = env.TicTacToeEnv(3, opponent=opp, agent_plays=game.X)
        e.reset()
        done = False
        for a in (0, 2, 3, 7, 8):
            if done:
                break
            _, _, done, _ = e.step(a)
        self.assertTrue(done)
        self.assertEqual(e.board.ply, 9)
        self.assertEqual(opp.seen, _expected(e.board))

    def test_self_play_without_opponent_still_steps(self):
        e = env.TicTacToeEnv(3)
        e.reset()
        for a in (0, 3, 1, 4):
            _, _, done, _ = e.step(a)
            self.assertFalse(done)
        _, r, done, _ = e.step(2)
        self.assertTrue(done)
        self.assertEqual(r, 1.0)

    def test_clone_tells_the_shared_opponent(self):
        opp = Recorder()
        e = env.TicTacToeEnv(3, opponent=opp, agent_plays=game.X)
        e.reset()
        c = e.clone()
        c.step(0)
        self.assertEqual(opp.seen, [(1, 0), (2, 8)])

    def test_batch_opponents_observe_their_own_game(self):
        batch = env.BatchEnv(3, 3, opponent_spec="random", seed=0)
        recs = [Recorder() for _ in batch.envs]
        for e, r in zip(batch.envs, recs):
            e.opponent = r
        batch.reset()
        for _ in range(3):
            batch.step([e.legal_actions()[0] for e in batch.envs])
        for e, r in zip(batch.envs, recs):
            self.assertEqual(r.seen, _expected(e.board))


class AlternateSelectTest(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)

    def test_first_game_is_x_and_second_is_o(self):
        batch = env.BatchEnv(3, 3, opponent_spec="fake_agent:LastLegal", seed=0)
        batch.reset()
        self.assertEqual([e.agent_plays for e in batch.envs], [game.X] * 3)
        dones = [False] * 3
        while not all(dones):
            _, _, d, _ = batch.step([e.legal_actions()[0] for e in batch.envs])
            dones = [a or b for a, b in zip(dones, d)]
        self.assertEqual([e.agent_plays for e in batch.envs], [game.O] * 3)

    def test_one_env_finishing_does_not_flip_others(self):
        batch = env.BatchEnv(
            2,
            3,
            opponent_spec="fake_agent:LastLegal",
            seed=0,
            illegal_move="lose",
        )
        batch.reset()
        sides = {0: [game.X], 1: [game.X]}
        e0, e1 = batch.envs
        # Env 0 ends five games by an illegal move; env 1 plays legally.
        while len(sides[0]) < 6:
            if e0.board.ply:
                a0 = e0.board.moves_played()[0]  # occupied: illegal
            else:
                a0 = 0
            _, _, d, _ = batch.step([a0, e1.legal_actions()[0]])
            for i in (0, 1):
                if d[i]:
                    sides[i].append(batch.envs[i].agent_plays)
        self.assertGreaterEqual(len(sides[1]), 2)
        flips0 = [a != b for a, b in zip(sides[0], sides[0][1:])]
        flips1 = [a != b for a, b in zip(sides[1], sides[1][1:])]
        self.assertTrue(all(flips0), sides[0])
        self.assertTrue(all(flips1), sides[1])

    def test_illegal_lose_counts_as_a_finished_game(self):
        batch = env.BatchEnv(
            1,
            3,
            opponent_spec="fake_agent:LastLegal",
            seed=0,
            illegal_move="lose",
        )
        batch.reset()
        batch.step([0])  # X 0, O 8
        _, r, d, info = batch.step([0])
        self.assertTrue(d[0])
        self.assertEqual(r[0], -1.0)
        self.assertTrue(info[0]["illegal"])
        self.assertEqual(batch.envs[0].agent_plays, game.O)

    def test_self_play_batch_runs_and_keeps_x(self):
        batch = env.BatchEnv(2, 3, seed=0)
        batch.reset()
        finished = 0
        for _ in range(30):
            _, _, d, _ = batch.step([e.legal_actions()[0] for e in batch.envs])
            finished += int(d.sum())
        self.assertGreater(finished, 2)
        self.assertEqual([e.agent_plays for e in batch.envs], [game.X] * 2)


if __name__ == "__main__":
    unittest.main()
