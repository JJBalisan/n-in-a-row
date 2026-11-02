"""Tests for n_in_a_row.harness.trainer, the distillation pipeline. NumPy.

Locally these skip without NumPy; with CI set they fail instead, like the
env tests. The example's own numbers are pinned in test_characterization.
"""

import os
import random
import sys
import tempfile
import unittest
from unittest import mock

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


class TestTrainer(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)
        from n_in_a_row.harness import trainer

        self.trainer = trainer

    def test_distilled_net_never_loses_to_random(self):
        policy = self.trainer.distil(3, walks=400, hidden=64, epochs=15)
        me = self.trainer.PolicyAgent(policy)
        res = arena.play_match(
            me, agents.RandomAgent(random.Random(0)), 40, rows=3
        )
        self.assertEqual(res.losses, 0)

    def test_example_and_library_give_the_same_arrays(self):
        import numpy as np

        from examples import learn_from_solver

        positions = self.trainer.collect_positions(3, 3, 3, 15, 1)
        self.assertEqual(
            [b.hash for b in positions],
            [
                b.hash
                for b in learn_from_solver.collect_positions(3, 3, 3, 15, 1)
            ],
        )
        lib = self.trainer.label(positions, None, True)
        with mock.patch("builtins.print"):
            ex = learn_from_solver.label(positions, None, True)
        for a, b in zip(lib, ex):
            np.testing.assert_array_equal(a, b)

    def test_custom_teacher_and_log(self):
        import numpy as np

        def corner(b):
            target = np.zeros(b.size, dtype=np.float32)
            legal = [m for m in (0, 2, 6, 8) if b.cells[m] == game.EMPTY]
            if not legal:
                return None
            for m in legal:
                target[m] = 1.0 / len(legal)
            return target, 0.0

        lines = []
        policy = self.trainer.distil(
            3,
            walks=200,
            target_fn=corner,
            hidden=32,
            epochs=20,
            augment=False,
            log=lines.append,
        )
        epochs = [line for line in lines if "epoch" in line]
        self.assertEqual(len(epochs), 20)
        self.assertTrue(epochs[-1].startswith("  epoch 20  cross-entropy"))
        self.assertIn("  labelled 500/", lines[0])
        me = self.trainer.PolicyAgent(policy)
        self.assertIn(me.select(game.Board()), (0, 2, 6, 8))

    def test_save_and_load_round_trip(self):
        import numpy as np

        policy = self.trainer.Policy(27, 4, 9, np.random.default_rng(0))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.npz")
            policy.save(path)
            back = self.trainer.Policy.load(path)
        for name in ("W1", "b1", "W2", "b2"):
            np.testing.assert_array_equal(
                getattr(policy, name), getattr(back, name)
            )


if __name__ == "__main__":
    unittest.main()
