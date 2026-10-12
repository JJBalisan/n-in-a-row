"""Stdlib-only tests:  python3 -m unittest discover -s tests"""

import contextlib
import io
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tictactoe import cli
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
            Board(3, 3, 4)  # k cannot exceed the longest side
        Board(3, 5, 4)  # but it fits along the long axis

    def test_rows_columns_diagonals_and_antidiagonals(self):
        for moves, winner in [
            ([0, 3, 1, 4, 2], X),  # top row
            ([0, 1, 3, 2, 6], X),  # left column
            ([0, 1, 4, 2, 8], X),  # main diagonal
            ([2, 0, 4, 1, 6], X),  # anti-diagonal
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
            b2.push(5)  # game is over

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
        b = Board.from_moves([0, 3, 1])  # X on a1,b1 -- c1 completes
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
        b = Board.from_moves([0, 4, 1])  # X threatens c1
        b.push(8)  # O ignores it
        r = solve(b)
        self.assertEqual(r.move, 2)
        self.assertTrue(r.value > WIN // 2)

    def test_blocks_the_loss(self):
        b = Board.from_moves([0, 4, 1])  # O must stop c1
        self.assertEqual(solve(b).move, 2)

    def test_prefers_the_faster_win(self):
        b = Board.from_moves([0, 3, 1, 4])  # X wins now at c1
        r = solve(b)
        self.assertEqual(r.move, 2)
        self.assertEqual(r.outcome, "win in 1")

    def test_depth_limit_is_flagged_inexact(self):
        r = Negamax(max_depth=2).search(Board(5, 5, 4))
        self.assertFalse(r.exact)
        self.assertIsNotNone(r.move)

    def test_warm_table_keeps_mate_distance(self):
        """A reused engine reports the same 'win in N' as a fresh one."""
        b = Board.from_moves([0, 4, 8, 2])
        engine = Negamax()
        engine.evaluate_moves(b)
        self.assertEqual(engine.search(b).outcome, Negamax().search(b).outcome)
        self.assertEqual(engine.search(b).outcome, "win in 2")

    def test_cli_solve_headline_agrees_with_its_table(self):
        """The solve headline matches the table row for its best move."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["solve", "--size", "3", "a1", "b2", "c3", "c1"])
        lines = out.getvalue().splitlines()
        headline = next(ln for ln in lines if " to move: " in ln)
        outcome = headline.split(" to move: ", 1)[1].strip()
        best = next(ln for ln in lines if ln.strip().startswith("best "))
        best_label = best.split()[1]
        rows = [ln.strip() for ln in lines if ln.startswith("  ")]
        table = dict(
            r.split(None, 1) for r in rows if not r.startswith("best ")
        )
        self.assertEqual(outcome, table[best_label])

    def test_table_is_not_reused_across_k(self):
        """Changing only k must not leak entries from the previous rules."""
        engine = Negamax()
        engine.evaluate_moves(Board(3, 3, 2))
        warm = engine.evaluate_moves(Board(3, 3, 3))
        fresh = Negamax().evaluate_moves(Board(3, 3, 3))
        self.assertEqual(warm[8].outcome, "draw")
        self.assertEqual(
            {m: r.outcome for m, r in warm.items()},
            {m: r.outcome for m, r in fresh.items()},
        )


class TestAgents(unittest.TestCase):
    def test_specs_load(self):
        for spec in (
            "random",
            "greedy",
            "perfect",
            "negamax:3",
            "time:0.05",
            "noisy:0.5:greedy",
        ):
            a = load_agent(spec, random.Random(0))
            self.assertIn(a.select(Board()), range(9))

    def test_unknown_spec_explains_itself(self):
        with self.assertRaises(ValueError):
            load_agent("nonsense")

    def test_greedy_wins_and_blocks(self):
        a = load_agent("greedy", random.Random(0))
        self.assertEqual(
            a.select(Board.from_moves([0, 3, 1, 4])), 2
        )  # X completes
        self.assertEqual(a.select(Board.from_moves([0, 4, 1])), 2)  # O blocks

    def test_perfect_never_loses_to_random(self):
        perfect = load_agent("perfect", random.Random(0))
        chaos = load_agent("random", random.Random(0))
        res = play_match(perfect, chaos, 40)
        self.assertEqual(res.losses, 0, "perfect play lost a game")

    def test_perfect_always_draws_itself(self):
        # perfect vs perfect is deterministic, so one game is the whole test.
        rec = play_game(
            load_agent("perfect", random.Random(0)),
            load_agent("perfect", random.Random(1)),
        )
        self.assertEqual(rec.winner, DRAW)

    def test_perfect_never_loses_to_noisy_perfect(self):
        """Varied games: a noisy copy strays, and perfect must never lose."""
        records = []
        res = play_match(
            load_agent("perfect", random.Random(0)),
            load_agent("noisy:0.3:perfect", random.Random(3)),
            6,
            records=records,
        )
        self.assertEqual(res.losses, 0, "perfect lost to noisy:0.3:perfect")
        self.assertGreater(
            len({tuple(r.moves) for r in records}),
            1,
            "every game was identical",
        )


class TestArena(unittest.TestCase):
    def test_match_accounting_adds_up(self):
        res = play_match(
            load_agent("greedy", random.Random(0)),
            load_agent("random", random.Random(0)),
            30,
        )
        self.assertEqual(res.games, 30)
        self.assertEqual(res.wins + res.draws + res.losses, 30)
        self.assertEqual(res.as_x["w"] + res.as_o["w"], res.wins)

    def test_replay_targets_match_the_result(self):
        rec = play_game(
            load_agent("greedy", random.Random(0)),
            load_agent("random", random.Random(2)),
        )
        steps = list(replay(rec))
        self.assertEqual(len(steps), len(rec.moves))
        board, _move, value = steps[0]
        self.assertEqual(board.ply, 0)
        self.assertEqual(value, rec.result_for(X))
        # Values alternate sign unless the game was drawn.
        if rec.winner != DRAW:
            self.assertEqual(steps[1][2], -value)


def _require_numpy(test):
    """Skips a NumPy test locally, but fails it when CI is set.

    A silent skip in CI would leave env.py untested behind a green run, so any
    non-empty CI variable (GitLab and GitHub both set CI=true) turns a missing
    NumPy into a failure. Install the test extra: pip install -e '.[test]'.

    Args:
        test: The running unittest.TestCase.
    """
    try:
        import numpy  # noqa: F401
    except ImportError:
        if os.environ.get("CI"):
            test.fail(
                "numpy is not installed but CI is set; "
                "install the test extra: pip install -e '.[test]'"
            )
        test.skipTest("numpy not installed (set CI=1 to make this a failure)")


class TestEnv(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)

    def test_observation_is_canonical(self):
        from tictactoe.env import encode

        b = Board.from_moves([4])  # X centre, O to move
        obs = encode(b)
        self.assertEqual(obs.shape, (3, 3, 3))
        self.assertEqual(obs[0].sum(), 0)  # O (to move) has no stones
        self.assertEqual(obs[1][1, 1], 1)  # opponent stone in the centre
        self.assertEqual(obs[2][0, 0], 0)  # side-to-move plane is O

    def test_step_reward_belongs_to_the_mover(self):
        from tictactoe.env import TicTacToeEnv

        env = TicTacToeEnv(3)
        env.reset()
        for a in (0, 3, 1, 4):
            _, r, done, _ = env.step(a)
            self.assertFalse(done)
        _, r, done, _ = env.step(2)  # X completes the top row
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

        self.assertEqual(
            len(
                dihedral(
                    np.zeros((3, 3, 3), np.float32),
                    np.zeros(9, np.float32),
                    3,
                    3,
                )
            ),
            8,
        )
        self.assertEqual(
            len(
                dihedral(
                    np.zeros((3, 2, 5), np.float32),
                    np.zeros(10, np.float32),
                    2,
                    5,
                )
            ),
            4,
        )

    def test_symmetries_preserve_the_move(self):
        """Each transform moves the policy target with the stone it marks."""
        import numpy as np

        from tictactoe.env import dihedral, encode

        for rows, cols, moves, target, n_variants in [
            (3, 3, [2, 3], 2, 8),  # X on c1 is the target, O on a2
            (2, 5, [1, 7], 1, 4),  # rectangle: no transposes
        ]:
            b = Board.from_moves(moves, rows, cols)
            policy = np.zeros(rows * cols, np.float32)
            policy[target] = 1.0
            obs = encode(b)
            mover_plane = 0 if b.cells[target] == b.to_move else 1
            self.assertEqual(obs[mover_plane].reshape(-1)[target], 1)
            seen = set()
            variants = dihedral(obs, policy, rows, cols)
            self.assertEqual(len(variants), n_variants)
            for obs_t, pol_t in variants:
                self.assertEqual(pol_t.sum(), 1.0)
                stone = obs_t[mover_plane].reshape(-1)
                self.assertEqual(stone.sum(), 1)
                self.assertEqual(int(np.argmax(stone)), int(np.argmax(pol_t)))
                seen.add(int(np.argmax(pol_t)))
            self.assertGreater(len(seen), 1, "no transform moved the target")


if __name__ == "__main__":
    unittest.main(verbosity=2)
