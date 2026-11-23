"""Checks for #36: each BatchEnv game alternates colours on its own."""

import os
import random
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))
sys.path.insert(0, _TESTS)


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


def _colour_sequences(batch, pick, games):
    """Plays until every env has started ``games`` games.

    Returns, per env, the side the trained agent held in each game, read
    from the public ``agent_plays`` after each auto-reset.
    """
    batch.reset()
    seqs = [[e.agent_plays] for e in batch.envs]
    for _ in range(games * 20):
        if min(len(s) for s in seqs) >= games:
            break
        actions = [pick(i, e) for i, e in enumerate(batch.envs)]
        _, _, dones, _ = batch.step(actions)
        for e, s, d in zip(batch.envs, seqs, dones):
            if d:
                s.append(e.agent_plays)
    return [s[:games] for s in seqs]


class BatchEnvAlternatesPerEnvTest(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)
        from n_in_a_row import env as rl_env

        self.rl_env = rl_env

    def _assert_flips_every_game(self, seqs):
        for i, seq in enumerate(seqs):
            flips = [a != b for a, b in zip(seq, seq[1:])]
            self.assertTrue(
                all(flips),
                f"env {i} did not swap sides after every game: {seq}",
            )

    def test_uneven_finishes_still_alternate_per_env(self):
        batch = self.rl_env.BatchEnv(4, 3, opponent_spec="random", seed=0)
        rngs = [random.Random(100 + i) for i in range(batch.n)]

        def pick(i, e):
            return rngs[i].choice(e.legal_actions())

        seqs = _colour_sequences(batch, pick, 16)
        self.assertTrue(all(len(s) == 16 for s in seqs))
        self._assert_flips_every_game(seqs)

    def test_games_finishing_in_the_same_step_each_swap(self):
        # Both sides play the lowest empty cell, so every env plays the same
        # first game and they all finish it on the same step.
        batch = self.rl_env.BatchEnv(
            2, 3, opponent_spec="fake_agent:FirstLegal", seed=0
        )

        def pick(i, e):
            return e.legal_actions()[0]

        seqs = _colour_sequences(batch, pick, 8)
        self.assertTrue(all(len(s) == 8 for s in seqs))
        self._assert_flips_every_game(seqs)

    def test_each_env_plays_both_colours_equally(self):
        batch = self.rl_env.BatchEnv(4, 3, opponent_spec="random", seed=1)
        rngs = [random.Random(200 + i) for i in range(batch.n)]

        def pick(i, e):
            return rngs[i].choice(e.legal_actions())

        seqs = _colour_sequences(batch, pick, 40)
        for i, seq in enumerate(seqs):
            self.assertEqual(
                seq.count(seq[0]),
                20,
                f"env {i} held one side in {seq.count(seq[0])} of 40 games",
            )


if __name__ == "__main__":
    unittest.main()
