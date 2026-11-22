"""Checks for #37: depth-limited solver targets must not depend on order.

A depth-limited search that keeps its transposition table between positions
labels later positions with what it learnt on earlier ones, so the targets
depend on the order the positions are labelled in. Exact labels (depth None)
are order-free already.
"""

import os
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import game


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


class TestDepthLimitedTargets(unittest.TestCase):
    DEPTH = 3
    #: Positions the order checks label; enough for 30+ to differ on base.
    SUBSET = 1000

    def setUp(self):
        _require_numpy(self)
        import numpy as np

        from n_in_a_row.harness import trainer

        self.np = np
        self.trainer = trainer
        self.positions = trainer.collect_positions(3, 3, 3, 3000, 0)

    def _fresh(self, board):
        return self.trainer.solver_targets(self.DEPTH)(board)

    def test_issue_example_matches_a_fresh_engine(self):
        # After b2, b1 the issue reports ['a2'] from the shared engine and
        # ['a1', 'c1', 'a3', 'c3'] from a fresh one.
        target = game.Board.from_moves([4, 1], 3, 3, 3)
        fn = self.trainer.solver_targets(self.DEPTH)
        got = None
        for p in self.positions:
            t, _ = fn(p)
            if p.cells == target.cells:
                got = sorted(int(m) for m in self.np.flatnonzero(t))
        self.assertIsNotNone(got, "b2, b1 is not in the walked positions")
        want_t, _ = self._fresh(target)
        want = sorted(int(m) for m in self.np.flatnonzero(want_t))
        self.assertEqual(want, [0, 2, 6, 8])
        self.assertEqual(got, want)

    def test_shared_teacher_equals_fresh_engine_per_position(self):
        fn = self.trainer.solver_targets(self.DEPTH)
        differ = []
        for i, p in enumerate(self.positions[: self.SUBSET]):
            t, v = fn(p)
            ft, fv = self._fresh(p)
            if not (self.np.array_equal(t, ft) and v == fv):
                differ.append(i)
        self.assertEqual(differ, [], f"{len(differ)} of {self.SUBSET} differ")

    def test_label_does_not_depend_on_order(self):
        pos = self.positions[: self.SUBSET]
        _, _, p_fwd, v_fwd = self.trainer.label(pos, self.DEPTH, False)
        _, _, p_rev, v_rev = self.trainer.label(pos[::-1], self.DEPTH, False)
        differ = sum(
            1
            for i in range(len(pos))
            if not self.np.array_equal(p_fwd[i], p_rev[-1 - i])
            or v_fwd[i] != v_rev[-1 - i]
        )
        self.assertEqual(differ, 0, f"{differ} of {len(pos)} targets differ")


if __name__ == "__main__":
    unittest.main()
