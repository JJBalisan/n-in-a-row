"""Checks for #52: zoo ground-truth logs carry the game of each entry.

The logs stay ``(ply, move)`` pairs that are never cleared (pinned in
tests/test_harness.py); the game goes beside them, one record per entry:
``Saboteur.throw_games`` beside ``throws`` and ``Sandbagger.deviation_games``
beside ``deviations``. The game is counted in ``reset()``, so every check
here gives each seat its own object (one object in both seats is reset
twice per game until #44 is fixed).

Stdlib only:  python3 -m unittest discover -s tests
"""

import os
import random
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents, arena, game
from n_in_a_row.harness import labels, zoo

LABELLER = labels.ExactLabeller()


class Recorder(agents.Agent):
    """Counts the hooks it is sent; plays the first legal move."""

    name = "recorder"

    def __init__(self):
        self.resets = 0
        self.observed = []

    def reset(self, board):
        self.resets += 1

    def observe(self, board, move):
        self.observed.append(move)

    def select(self, board):
        return board.legal_moves()[0]


def _saboteur(seed=0):
    return zoo.Saboteur(
        agents.NegamaxAgent(rng=random.Random(seed)),
        1.0,
        labeller=LABELLER,
        rng=random.Random(seed),
    )


def _sandbagger(seed=0, mode="draw"):
    return zoo.Sandbagger(LABELLER, mode=mode, rng=random.Random(seed))


def _match(agent, log, games, seed=0):
    """Plays ``games`` games against a random opponent, alternating seats.

    Returns the true game number (0-based) of every new entry in ``log``.
    """
    opponent = agents.RandomAgent(random.Random(seed))
    truth = []
    for g in range(games):
        before = len(log)
        if g % 2 == 0:
            arena.play_game(agent, opponent)
        else:
            arena.play_game(opponent, agent)
        truth.extend([g] * (len(log) - before))
    return truth


class GameIndexCase(unittest.TestCase):
    def assert_joins(self, record, truth):
        """Fails unless ``record`` groups the entries as ``truth`` does."""
        self.assertIsInstance(
            record, list, "no per-entry game record beside the log"
        )
        self.assertEqual(len(record), len(truth), "one record per entry")
        for i in range(len(truth) - 1):
            if truth[i] == truth[i + 1]:
                self.assertEqual(
                    record[i], record[i + 1], f"entries {i}, {i + 1}"
                )
            else:
                self.assertLess(
                    record[i], record[i + 1], f"entries {i}, {i + 1}"
                )

    def assert_not_vacuous(self, truth):
        games = set(truth)
        self.assertGreaterEqual(len(games), 2, truth)
        self.assertTrue(any(truth.count(g) >= 2 for g in games), truth)


class TestSaboteurThrowGames(GameIndexCase):
    def test_throws_join_to_the_games_of_a_match(self):
        sab = _saboteur()
        truth = _match(sab, sab.throws, 4)
        self.assert_not_vacuous(truth)
        self.assert_joins(getattr(sab, "throw_games", None), truth)

    def test_throws_stay_uncleared_ply_move_pairs(self):
        sab = _saboteur()
        truth = _match(sab, sab.throws, 3)
        self.assertEqual(len(sab.throws), len(truth))
        self.assertGreater(len(set(truth)), 1)
        for entry in sab.throws:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)
        self.assertIsInstance(getattr(sab, "throw_games", None), list)

    def test_throws_before_the_first_reset_get_a_game(self):
        sab = _saboteur()
        board = game.Board.from_moves([0])
        sab.select(board)
        sab.select(board)
        self.assertEqual(len(sab.throws), 2)
        sab.reset(game.Board())
        sab.select(board)
        self.assertEqual(len(sab.throws), 3)
        self.assert_joins(getattr(sab, "throw_games", None), [0, 0, 1])

    def test_reset_still_reaches_the_base(self):
        base = Recorder()
        sab = zoo.Saboteur(base, 1.0, labeller=LABELLER)
        sab.reset(game.Board())
        sab.observe(game.Board.from_moves([4]), 4)
        self.assertEqual((base.resets, base.observed), (1, [4]))
        self.assertIsInstance(getattr(sab, "throw_games", None), list)

    def test_two_saboteurs_keep_their_own_records(self):
        one, two = _saboteur(0), _saboteur(1)
        truth = _match(one, one.throws, 2)
        self.assertEqual(getattr(two, "throw_games", None), [])
        self.assert_joins(getattr(one, "throw_games", None), truth)


class TestSandbaggerDeviationGames(GameIndexCase):
    def test_deviations_join_to_the_games_of_a_match(self):
        sb = _sandbagger()
        truth = _match(sb, sb.deviations, 8)
        self.assert_not_vacuous(truth)
        self.assert_joins(getattr(sb, "deviation_games", None), truth)

    def test_slow_deviations_join_to_the_games_of_a_match(self):
        sb = _sandbagger(1, mode="slow")
        truth = _match(sb, sb.deviations, 8, seed=1)
        self.assert_not_vacuous(truth)
        self.assert_joins(getattr(sb, "deviation_games", None), truth)

    def test_deviations_stay_uncleared_ply_move_pairs(self):
        sb = _sandbagger()
        truth = _match(sb, sb.deviations, 8)
        self.assertEqual(len(sb.deviations), len(truth))
        self.assertGreater(len(set(truth)), 1)
        for entry in sb.deviations:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)
        self.assertIsInstance(getattr(sb, "deviation_games", None), list)

    def test_reset_still_reaches_the_fallback(self):
        fallback = Recorder()
        sb = zoo.Sandbagger(LABELLER, fallback=fallback)
        sb.reset(game.Board())
        sb.observe(game.Board.from_moves([4]), 4)
        self.assertEqual((fallback.resets, fallback.observed), (1, [4]))
        self.assertIsInstance(getattr(sb, "deviation_games", None), list)

    def test_two_sandbaggers_keep_their_own_records(self):
        one, two = _sandbagger(0), _sandbagger(1)
        truth = _match(one, one.deviations, 8)
        self.assertEqual(getattr(two, "deviation_games", None), [])
        self.assert_joins(getattr(one, "deviation_games", None), truth)


if __name__ == "__main__":
    unittest.main()
