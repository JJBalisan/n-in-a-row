"""Selector's adversarial checks for #37 and #47 (sweep 22 Nov 2026).

#37: depth-limited solver targets must equal a fresh engine's on every
position, whatever was labelled before, at several depths and board
sizes; exact labels (depth None) must not change; the teacher must not
change the board it labels.

#47: a PolicyAgent whose weights do not fit the board must raise a
ValueError naming the board, never a raw NumPy error, on every path
(greedy, sampling, a rectangle, a mismatched output layer), and must
still play on a board that fits after such an error.
"""

import os
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents, arena, game


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


class TestSelectDepthTargets(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)
        import numpy as np

        from n_in_a_row.harness import trainer

        self.np = np
        self.trainer = trainer

    def _key(self, out):
        if out is None:
            return None
        t, v = out
        return (tuple(float(x) for x in t), float(v))

    def _assert_order_free(self, positions, depth):
        fn = self.trainer.solver_targets(depth)
        shared = [self._key(fn(p)) for p in positions]
        fresh = [
            self._key(self.trainer.solver_targets(depth)(p)) for p in positions
        ]
        bad = [i for i, (a, b) in enumerate(zip(shared, fresh)) if a != b]
        self.assertEqual(bad, [], f"depth {depth}: {len(bad)} differ")

    def test_depths_2_and_4_on_3x3(self):
        positions = self.trainer.collect_positions(3, 3, 3, 3000, 1)[:200]
        for depth in (2, 4):
            with self.subTest(depth=depth):
                self._assert_order_free(positions, depth)

    def test_depth_2_on_4x4_k3(self):
        positions = self.trainer.collect_positions(4, 4, 3, 40, 2)[:15]
        self._assert_order_free(positions, 2)

    def test_same_position_twice_gives_same_target(self):
        positions = self.trainer.collect_positions(3, 3, 3, 3000, 0)[:300]
        fn = self.trainer.solver_targets(3)
        first = [self._key(fn(p)) for p in positions]
        second = [self._key(fn(p)) for p in positions]
        self.assertEqual(first, second)

    def test_exact_labels_unchanged(self):
        positions = self.trainer.collect_positions(3, 3, 3, 3000, 0)[:15]
        self._assert_order_free(positions, None)

    def test_teacher_leaves_the_board_alone(self):
        positions = self.trainer.collect_positions(3, 3, 3, 200, 3)[:50]
        fn = self.trainer.solver_targets(3)
        for p in positions:
            before = (list(p.cells), p.hash, len(p.legal_moves()))
            fn(p)
            self.assertEqual(
                before, (list(p.cells), p.hash, len(p.legal_moves()))
            )

    def test_label_reverse_order_matches(self):
        positions = self.trainer.collect_positions(3, 3, 3, 3000, 0)[:300]
        fwd = self.trainer.label(positions, depth=3, augment=False)
        rev = self.trainer.label(positions[::-1], depth=3, augment=False)
        self.np.testing.assert_array_equal(fwd[2], rev[2][::-1])
        self.np.testing.assert_array_equal(fwd[3], rev[3][::-1])


class TestSelectShapeCheck(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)
        import numpy as np

        from n_in_a_row.harness import trainer

        self.np = np
        self.trainer = trainer

    def _agent(self, n_in=27, n_out=9, greedy=True):
        policy = self.trainer.Policy(
            n_in, 8, n_out, self.np.random.default_rng(0)
        )
        return self.trainer.PolicyAgent(policy, greedy=greedy)

    def _assert_clear(self, agent, board):
        with self.assertRaises(ValueError) as cm:
            agent.reset(board)
            agent.select(board)
        msg = str(cm.exception)
        for raw in ("matmul", "broadcast", "operand"):
            self.assertNotIn(raw, msg, f"raw NumPy error: {msg}")
        self.assertIn("board", msg.lower(), f"names no board: {msg}")

    def test_sampling_agent_on_4x4(self):
        self._assert_clear(self._agent(greedy=False), game.Board(4, 4, 3))

    def test_rectangle_3x4(self):
        self._assert_clear(self._agent(), game.Board(3, 4, 3))

    def test_smaller_board_2x2(self):
        self._assert_clear(self._agent(), game.Board(2, 2, 2))

    def test_output_layer_mismatch(self):
        # Input fits a 3x3 board but the net has 10 outputs, not 9.
        self._assert_clear(self._agent(n_out=10), game.Board(3, 3, 3))

    def test_plays_3x3_after_a_4x4_error(self):
        agent = self._agent()
        with self.assertRaises(ValueError):
            b4 = game.Board(4, 4, 3)
            agent.reset(b4)
            agent.select(b4)
        b3 = game.Board(3, 3, 3)
        agent.reset(b3)
        move = agent.select(b3)
        self.assertIn(move, b3.legal_moves())

    def test_3x3_weights_any_k(self):
        for k in (2, 3):
            with self.subTest(k=k):
                me = agents.load_agent("examples.learn_from_solver:agent")
                them = agents.load_agent("random")
                res = arena.play_match(me, them, 2, rows=3, cols=3, k=k)
                self.assertIn("policy", res.summary("policy", "random"))


if __name__ == "__main__":
    unittest.main()
