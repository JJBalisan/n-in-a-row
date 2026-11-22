"""Checks for #47: 3x3 weights on another board size must fail clearly.

The example factories load weights trained on 3x3. On a 4x4 board the
first move used to die inside NumPy's matmul; it must instead raise a
ValueError that says the weights do not fit the board. Where the check
sits (reset or select) and the exact wording are left to the fix.
"""

import os
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents, arena, game

FACTORIES = (
    "examples.train_policy:agent",
    "examples.learn_from_solver:agent",
)


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


class TestWeightShapeCheck(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)

    def _assert_clear_error(self, err):
        msg = str(err)
        self.assertNotIn("matmul", msg, f"raw NumPy error: {msg}")
        self.assertIn("board", msg.lower(), f"message names no board: {msg}")

    def test_first_move_on_4x4_raises_a_clear_value_error(self):
        for spec in FACTORIES:
            with self.subTest(spec=spec):
                agent = agents.load_agent(spec)
                board = game.Board(4, 4, 3)
                with self.assertRaises(ValueError) as cm:
                    agent.reset(board)
                    agent.select(board)
                self._assert_clear_error(cm.exception)

    def test_match_on_4x4_raises_a_clear_value_error(self):
        for spec in FACTORIES:
            with self.subTest(spec=spec):
                me = agents.load_agent(spec)
                them = agents.load_agent("random")
                with self.assertRaises(ValueError) as cm:
                    arena.play_match(me, them, 2, rows=4, cols=4, k=3)
                self._assert_clear_error(cm.exception)

    def test_3x3_still_plays(self):
        for spec in FACTORIES:
            with self.subTest(spec=spec):
                me = agents.load_agent(spec)
                them = agents.load_agent("random")
                res = arena.play_match(me, them, 2, rows=3, cols=3, k=3)
                self.assertIn("policy", res.summary("policy", "random"))

    def test_policy_agent_names_the_mismatch(self):
        import numpy as np

        from n_in_a_row.harness import trainer

        policy = trainer.Policy(27, 8, 9, np.random.default_rng(0))
        agent = trainer.PolicyAgent(policy)
        board = game.Board(4, 4, 3)
        with self.assertRaises(ValueError) as cm:
            agent.reset(board)
            agent.select(board)
        self._assert_clear_error(cm.exception)


if __name__ == "__main__":
    unittest.main()
