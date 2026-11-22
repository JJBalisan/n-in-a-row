"""Selector's adversarial tests for #31, #39 and #62, lane n-search."""

import os
import random
import sys
import unittest
from unittest import mock

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents, game, search


def _cut(engine, board):
    """Runs engine.search with a clock that reads the node count."""
    with mock.patch.object(search.time, "perf_counter", lambda: engine.nodes):
        return engine.search(board)


class TestWonBoard(unittest.TestCase):
    """#31 edge cases."""

    def test_win_on_the_last_cell_is_not_a_draw(self):
        # X wins with the ninth stone: the board is full AND won.
        b = game.Board.from_moves([0, 1, 2, 3, 4, 5, 7, 6, 8], 3, 3, 3)
        self.assertEqual(b.winner, game.X)
        r = search.Negamax().search(b)
        self.assertTrue(r.exact)
        self.assertIsNone(r.move)
        self.assertNotEqual(r.outcome, "draw")
        self.assertLess(r.value, -(search.WIN // 2))

    def test_won_value_matches_terminal_scoring(self):
        # _negamax scores a finished win as -(WIN - plies), 0 plies here;
        # evaluate_moves scores the winning move as WIN - 1 = shift_mate.
        b = game.Board.from_moves([0, 1, 3, 4, 6], 3, 3, 3)
        r = search.Negamax().search(b)
        self.assertEqual(r.value, -search.WIN)
        self.assertEqual(search.shift_mate(-r.value), search.WIN - 1)

    def test_won_board_left_unchanged_and_time_limit_ok(self):
        b = game.Board.from_moves([0, 1, 3, 4, 6], 3, 3, 3)
        before = (list(b.cells), b.ply, b.hash, b.winner)
        r = search.Negamax(time_limit=5.0).search(b)
        self.assertTrue(r.exact)
        self.assertLess(r.value, -(search.WIN // 2))
        self.assertEqual((list(b.cells), b.ply, b.hash, b.winner), before)

    def test_evaluate_moves_on_won_board_is_empty(self):
        b = game.Board.from_moves([0, 1, 3, 4, 6], 3, 3, 3)
        self.assertEqual(search.Negamax().evaluate_moves(b), {})

    def test_live_positions_unchanged(self):
        # Regression: an empty 3x3 board is still a solved draw.
        r = search.Negamax().search(game.Board(3, 3, 3))
        self.assertEqual((r.exact, r.outcome), (True, "draw"))
        self.assertIsNotNone(r.move)


class TestDepthValidation(unittest.TestCase):
    """#39 edge cases."""

    def test_depth_one_searches(self):
        r = search.Negamax(max_depth=1).search(game.Board(3, 3, 3))
        self.assertIsNotNone(r.move)
        self.assertEqual(r.depth, 1)
        self.assertNotEqual(r.value, -float("inf"))

    def test_bad_depth_rejected_with_time_limit_too(self):
        with self.assertRaises(ValueError):
            search.Negamax(max_depth=0, time_limit=1.0)

    def test_valid_specs_still_load(self):
        for spec in ("negamax:1", "negamax:3"):
            agent = agents.load_agent(spec, random.Random(0))
            b = game.Board(3, 3, 3)
            self.assertIn(agent.select(b), b.legal_moves())

    def test_rejected_depth_does_not_fall_back_to_random(self):
        # The issue: negamax:0 silently played random moves.
        with self.assertRaises(ValueError):
            agents.load_agent("negamax:0", random.Random(0))


class TestCutPassWin(unittest.TestCase):
    """#62 edge cases."""

    def test_win_in_first_pass_cut_is_exact(self):
        # 33x33, k=3: more than 1,024 root moves, so pass 1 is cut at node
        # 1,024 after its first (winning, ordered-first) move. No pass
        # completed, yet the win is proven.
        b = game.Board(33, 33, 3)
        for m in (0, 1088, 1, 1087):
            b.push(m)
        engine = search.Negamax(time_limit=1023)
        r = _cut(engine, b)
        self.assertLess(r.nodes, len(b.legal_moves()) + 1)
        self.assertEqual(r.move, 2)
        self.assertTrue(r.exact)
        self.assertEqual(r.outcome, "win in 1")

    def test_kept_win_is_a_real_win(self):
        # Play the kept move, then search the reply exactly: the opponent
        # must be lost (positions from the verifier's checks).
        for moves, nodes in (([0, 10, 15, 1], 3071), ([7, 11], 1023)):
            b = game.Board.from_moves(moves, 4, 4, 3)
            r = _cut(search.Negamax(time_limit=nodes), b)
            self.assertTrue(r.exact, moves)
            b.push(r.move)
            reply = search.Negamax().search(b)
            self.assertTrue(reply.exact, moves)
            self.assertLess(reply.value, -(search.WIN // 2), moves)
            self.assertEqual(search.shift_mate(-reply.value), r.value, moves)

    def test_cut_search_leaves_board_unchanged(self):
        b = game.Board.from_moves([0, 10, 15, 1], 4, 4, 3)
        before = (list(b.cells), b.ply, b.hash)
        _cut(search.Negamax(time_limit=3071), b)
        self.assertEqual((list(b.cells), b.ply, b.hash), before)

    def test_reused_engine_after_cut_win(self):
        # A warm table after a kept cut-pass win must not change the
        # exact answer of a full search on the same position.
        b = game.Board.from_moves([7, 11], 4, 4, 3)
        engine = search.Negamax(time_limit=1023)
        _cut(engine, b)
        engine.time_limit = None
        warm = engine.search(b)
        cold = search.Negamax().search(b)
        self.assertEqual((warm.exact, warm.outcome), (cold.exact, cold.outcome))


if __name__ == "__main__":
    unittest.main()
