"""Stdlib-only tests:  python3 -m unittest discover -s tests"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tictactoe.agents import load_agent
from tictactoe.arena import play_game, play_match, replay
from tictactoe.game import DRAW, Board, InvalidMove, O, X, windows
from tictactoe.search import WIN, Negamax, solve


class TestBoard(unittest.TestCase):
    def test_default_is_classic(self):
        b = Board()
        self.assertEqual(b.spec, (3, 3, 3))
        self.assertEqual(len(b.legal_moves()), 9)
        self.assertIs(b.winner, None)

    def test_rectangles_and_k(self):
        b = Board(4, 7, 4)
        self.assertEqual(b.spec, (4, 7, 4))
        self.assertEqual(b.size, 28)
        with self.assertRaises(ValueError):
            Board(3, 3, 4)          # k cannot exceed the longest side
        Board(3, 5, 4)              # but it fits along the long axis

    def test_rows_columns_diagonals_and_antidiagonals(self):
        for moves, winner in [
            ([0, 3, 1, 4, 2], X),          # top row
            ([0, 1, 3, 2, 6], X),          # left column
            ([0, 1, 4, 2, 8], X),          # main diagonal
            ([2, 0, 4, 1, 6], X),          # anti-diagonal
        ]:
            b = Board.from_moves(moves)
            self.assertEqual(b.winner, winner, moves)
            self.assertEqual(len(b.win_line), 3)

    def test_full_board_is_a_draw(self):
        b = Board.from_moves([0, 1, 2, 4, 3, 5, 7, 6, 8])
        self.assertEqual(b.winner, DRAW)
        self.assertIsNone(b.win_line)
        self.assertEqual(b.result_for(X), 0.0)

    def test_illegal_moves_rejected(self):
        b = Board()
        b.push(0)
        with self.assertRaises(InvalidMove):
            b.push(0)
        with self.assertRaises(InvalidMove):
            b.push(99)
        b2 = Board.from_moves([0, 3, 1, 4, 2])
        with self.assertRaises(InvalidMove):
            b2.push(5)              # game is over

    def test_push_pop_restores_everything(self):
        rng = random.Random(1)
        for _ in range(50):
            b = Board(4, 5, 4)
            snapshots = []
            while not b.is_terminal:
                snapshots.append((list(b.cells), b.to_move, b.hash, b.winner))
                b.push(rng.choice(b.legal_moves()))
            while snapshots:
                b.pop()
                cells, to_move, h, winner = snapshots.pop()
                self.assertEqual(b.cells, cells)
                self.assertEqual(b.to_move, to_move)
                self.assertEqual(b.hash, h)
                self.assertEqual(b.winner, winner)

    def test_hash_is_position_not_path(self):
        a = Board.from_moves([0, 4, 8])
        b = Board.from_moves([8, 4, 0])
        self.assertEqual(a.hash, b.hash)
        self.assertNotEqual(a.hash, Board.from_moves([0, 4, 7]).hash)

    def test_labels_round_trip(self):
        b = Board(3, 30, 3)
        for i in range(b.size):
            self.assertEqual(b.parse_move(b.label(i)), i)
        self.assertEqual(b.parse_move("2,3"), b.index(1, 2))
        self.assertEqual(b.parse_move("7"), 7)

    def test_window_count(self):
        # 3x3: 3 rows + 3 cols + 2 diagonals
        self.assertEqual(len(windows(3, 3, 3)), 8)
        self.assertEqual(len(windows(4, 4, 4)), 10)

    def test_winning_moves(self):
        b = Board.from_moves([0, 3, 1])       # X on a1,b1 -- c1 completes
        self.assertIn(2, b.winning_moves(X))
        self.assertNotIn(2, b.winning_moves(O))


class TestSearch(unittest.TestCase):
    def test_empty_3x3_is_a_draw(self):
        r = solve(Board())
        self.assertTrue(r.exact)
        self.assertEqual(r.outcome, "draw")

    def test_every_opening_draws(self):
        engine = Negamax()
        for m, r in engine.evaluate_moves(Board()).items():
            self.assertEqual(r.outcome, "draw", "opening %d" % m)

    def test_takes_the_win(self):
        b = Board.from_moves([0, 4, 1])       # X threatens c1
        b.push(8)                             # O ignores it
        r = solve(b)
        self.assertEqual(r.move, 2)
        self.assertTrue(r.value > WIN // 2)

    def test_blocks_the_loss(self):
        b = Board.from_moves([0, 4, 1])       # O must stop c1
        self.assertEqual(solve(b).move, 2)

    def test_prefers_the_faster_win(self):
        b = Board.from_moves([0, 3, 1, 4])    # X wins now at c1
        r = solve(b)
        self.assertEqual(r.move, 2)
        self.assertEqual(r.outcome, "win in 1")

    def test_depth_limit_is_flagged_inexact(self):
        r = Negamax(max_depth=2).search(Board(5, 5, 4))
        self.assertFalse(r.exact)
        self.assertIsNotNone(r.move)


class TestAgents(unittest.TestCase):
    def test_specs_load(self):
        for spec in ("random", "greedy", "perfect", "negamax:3", "time:0.05",
                     "noisy:0.5:greedy"):
            a = load_agent(spec, random.Random(0))
            self.assertIn(a.select(Board()), range(9))

    def test_unknown_spec_explains_itself(self):
        with self.assertRaises(ValueError):
            load_agent("nonsense")

    def test_greedy_wins_and_blocks(self):
        a = load_agent("greedy", random.Random(0))
        self.assertEqual(a.select(Board.from_moves([0, 3, 1, 4])), 2)   # X completes
        self.assertEqual(a.select(Board.from_moves([0, 4, 1])), 2)      # O blocks

    def test_perfect_never_loses_to_random(self):
        perfect = load_agent("perfect", random.Random(0))
        chaos = load_agent("random", random.Random(0))
        res = play_match(perfect, chaos, 40)
        self.assertEqual(res.losses, 0, "perfect play lost a game")

    def test_perfect_always_draws_itself(self):
        for _ in range(6):
            rec = play_game(load_agent("perfect", random.Random(0)),
                            load_agent("perfect", random.Random(1)))
            self.assertEqual(rec.winner, DRAW)


class TestArena(unittest.TestCase):
    def test_match_accounting_adds_up(self):
        res = play_match(load_agent("greedy", random.Random(0)),
                         load_agent("random", random.Random(0)), 30)
        self.assertEqual(res.games, 30)
        self.assertEqual(res.wins + res.draws + res.losses, 30)
        self.assertEqual(res.as_x["w"] + res.as_o["w"], res.wins)

    def test_replay_targets_match_the_result(self):
        rec = play_game(load_agent("greedy", random.Random(0)),
                        load_agent("random", random.Random(2)))
        steps = list(replay(rec))
        self.assertEqual(len(steps), len(rec.moves))
        board, _move, value = steps[0]
        self.assertEqual(board.ply, 0)
        self.assertEqual(value, rec.result_for(X))
        # Values alternate sign unless the game was drawn.
        if rec.winner != DRAW:
            self.assertEqual(steps[1][2], -value)


class TestEnv(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

    def test_observation_is_canonical(self):
        from tictactoe.env import encode
        b = Board.from_moves([4])            # X centre, O to move
        obs = encode(b)
        self.assertEqual(obs.shape, (3, 3, 3))
        self.assertEqual(obs[0].sum(), 0)    # O (to move) has no stones
        self.assertEqual(obs[1][1, 1], 1)    # opponent stone in the centre
        self.assertEqual(obs[2][0, 0], 0)    # side-to-move plane is O

    def test_step_reward_belongs_to_the_mover(self):
        from tictactoe.env import TicTacToeEnv
        env = TicTacToeEnv(3)
        env.reset()
        for a in (0, 3, 1, 4):
            _, r, done, _ = env.step(a)
            self.assertFalse(done)
        _, r, done, _ = env.step(2)           # X completes the top row
        self.assertTrue(done)
        self.assertEqual(r, 1.0)

    def test_illegal_move_can_forfeit(self):
        from tictactoe.env import TicTacToeEnv
        env = TicTacToeEnv(3, illegal_move="lose")
        env.reset()
        env.step(0)
        _, r, done, info = env.step(0)
        self.assertTrue(done and info["illegal"])
        self.assertEqual(r, -1.0)

    def test_symmetry_counts(self):
        import numpy as np

        from tictactoe.env import dihedral
        self.assertEqual(len(dihedral(np.zeros((3, 3, 3), np.float32),
                                      np.zeros(9, np.float32), 3, 3)), 8)
        self.assertEqual(len(dihedral(np.zeros((3, 2, 5), np.float32),
                                      np.zeros(10, np.float32), 2, 5)), 4)

    def test_symmetries_preserve_the_move(self):
        import numpy as np

        from tictactoe.env import dihedral, encode
        b = Board.from_moves([0, 4])
        policy = np.zeros(9, np.float32)
        policy[2] = 1.0
        for obs_t, pol_t in dihedral(encode(b), policy, 3, 3):
            self.assertEqual(pol_t.sum(), 1.0)
            self.assertEqual(obs_t.shape, (3, 3, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
