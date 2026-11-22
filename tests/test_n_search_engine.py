"""Checks for #31, #39 and #62 (search.py), lane n-search."""

import os
import random
import sys
import unittest
from unittest import mock

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents, game, search


def _cut_search(board, nodes):
    """Searches with a clock that reads the node count.

    The deadline is checked every 1,024 nodes, so a limit of 1,024n - 1
    cuts the search at node 1,024n, the same on every machine.
    """
    engine = search.Negamax(time_limit=nodes)
    with mock.patch.object(search.time, "perf_counter", lambda: engine.nodes):
        return engine.search(board)


class TestFinishedGame(unittest.TestCase):
    """#31: a won position must not be reported as a draw."""

    def test_full_board_draw_is_still_a_draw(self):
        # Guard: a real draw keeps its exact draw result.
        b = game.Board.from_moves([0, 4, 8, 1, 7, 6, 2, 5, 3], 3, 3, 3)
        self.assertEqual(b.winner, game.DRAW)
        r = search.Negamax().search(b)
        self.assertEqual((r.move, r.exact, r.outcome), (None, True, "draw"))

    def test_won_position_is_a_loss_for_the_side_to_move(self):
        cases = (
            # X completes the a-column; O to move.
            ([0, 1, 3, 4, 6], 3, 3, 3),
            # O completes the middle column; X to move.
            ([0, 1, 2, 4, 3, 7], 3, 3, 3),
            # 4x4, k=3, X completes a diagonal; O to move.
            ([0, 3, 5, 12, 10], 4, 4, 3),
        )
        for moves, rows, cols, k in cases:
            b = game.Board.from_moves(moves, rows, cols, k)
            self.assertIsNotNone(b.winner, moves)
            self.assertNotEqual(b.winner, game.DRAW, moves)
            self.assertEqual(b.legal_moves(), [], moves)
            for r in (
                search.Negamax().search(b),
                search.Negamax(max_depth=3).search(b),
                search.solve(b),
            ):
                self.assertNotEqual(r.outcome, "draw", moves)
                self.assertTrue(r.exact, moves)
                self.assertIsNone(r.move, moves)
                # Score is from the side to move, who has lost.
                self.assertLess(r.value, -(search.WIN // 2), moves)


class TestDepthBelowOne(unittest.TestCase):
    """#39: a depth below 1 is rejected when the engine is built."""

    def test_valid_depths_still_build(self):
        # Guard: None and positive depths are untouched.
        for depth in (None, 1, 4):
            search.Negamax(max_depth=depth)

    def test_engine_rejects_depth_below_one(self):
        for depth in (0, -1, -3):
            with self.assertRaises(ValueError, msg=depth):
                search.Negamax(max_depth=depth)
            with self.assertRaises(ValueError, msg=depth):
                search.solve(game.Board(3, 3, 3), max_depth=depth)

    def test_negamax_spec_rejects_depth_below_one(self):
        for spec in ("negamax:0", "negamax:-3"):
            with self.assertRaises(ValueError, msg=spec):
                agents.load_agent(spec, random.Random(0))


class TestCutPassProvenWin(unittest.TestCase):
    """#62: a proven win from a time-cut pass is kept and reported."""

    def test_cut_pass_without_a_proven_win_stays_discarded(self):
        # Guard: a cut pass's negative mate covers only some root moves,
        # so it is no proof (4x5, k=4, a draw; pass 5 is cut holding
        # b2 = -999996).
        b = game.Board.from_moves([10, 18, 8, 7, 1, 19, 2], 4, 5, 4)
        r = _cut_search(b, 2047)
        self.assertFalse(r.exact)
        self.assertLess(abs(r.value), search.WIN // 2)
        self.assertNotEqual(b.label(r.move), "b2")
        # Guard: nor is a higher heuristic score (5x5, k=4; pass 3 is cut
        # holding c2 = 23.0 over the completed pass 2's b5 = -11.0).
        b = game.Board.from_moves([23, 11, 24], 5, 5, 4)
        r = _cut_search(b, 1023)
        self.assertFalse(r.exact)
        whole = search.Negamax(max_depth=r.depth).search(b)
        self.assertEqual((r.move, r.value), (whole.move, whole.value))

    def test_proven_win_in_cut_pass_is_kept(self):
        cases = (
            # Pass 7 is cut after proving a2 = 999993 (win in 4).
            ([0, 10, 15, 1], 4, 4, 3, 3071, "a2"),
            # Pass 5 is cut after proving c2 = 999995 (win in 3).
            ([7, 11], 4, 4, 3, 1023, "c2"),
        )
        for moves, rows, cols, k, nodes, proved in cases:
            b = game.Board.from_moves(moves, rows, cols, k)
            truth = search.Negamax().search(b)
            self.assertTrue(truth.exact and truth.value > 0, moves)
            r = _cut_search(b, nodes)
            self.assertLess(r.nodes, truth.nodes, moves)  # really cut
            self.assertEqual(b.label(r.move), proved, moves)
            self.assertTrue(r.exact, moves)
            self.assertGreater(r.value, search.WIN // 2, moves)
            self.assertEqual(r.outcome, truth.outcome, moves)


if __name__ == "__main__":
    unittest.main()
