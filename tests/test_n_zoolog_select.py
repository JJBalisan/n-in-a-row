"""Selector checks for #52, written from the issue before reading a fix.

They add neighbours of the verifier's checks: whole matches through
``arena.play_match``, agents built by ``zoo.load``, a wrapper that forwards
``reset``, games with no entries, the fallback path, and a random mix of
``select`` and ``reset`` calls. Each seat is its own object (#44).

Stdlib only (the zoo does not need NumPy):
    python3 -m unittest tests.test_n_zoolog_select
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


def _groups_like(record, truth):
    """True when ``record`` groups entries exactly as ``truth`` does."""
    if not isinstance(record, list) or len(record) != len(truth):
        return False
    for i in range(len(truth) - 1):
        same = truth[i] == truth[i + 1]
        if same and record[i] != record[i + 1]:
            return False
        if not same and not record[i] < record[i + 1]:
            return False
    return True


def _match_truth(agent, opponent, log, games):
    """Plays a ``play_match``; returns the true game of each new entry."""
    truth = []
    sizes = [len(log)]

    def progress(done, total, result):
        truth.extend([done - 1] * (len(log) - sizes[-1]))
        sizes.append(len(log))

    arena.play_match(agent, opponent, games, progress=progress)
    return truth


class TestPlayMatch(unittest.TestCase):
    def test_saboteur_throws_join_through_play_match(self):
        sab = zoo.Saboteur(
            agents.NegamaxAgent(rng=random.Random(3)),
            1.0,
            labeller=LABELLER,
            rng=random.Random(3),
        )
        opp = agents.RandomAgent(random.Random(4))
        truth = _match_truth(sab, opp, sab.throws, 6)
        self.assertGreaterEqual(len(set(truth)), 2, truth)
        self.assertTrue(_groups_like(sab.throw_games, truth), sab.throw_games)

    def test_sandbagger_deviations_join_through_play_match(self):
        sb = zoo.Sandbagger(LABELLER, "draw", rng=random.Random(5))
        opp = agents.RandomAgent(random.Random(6))
        truth = _match_truth(sb, opp, sb.deviations, 10)
        self.assertGreaterEqual(len(set(truth)), 2, truth)
        self.assertTrue(
            _groups_like(sb.deviation_games, truth), sb.deviation_games
        )

    def test_records_are_ints(self):
        sab = zoo.Saboteur(
            agents.NegamaxAgent(rng=random.Random(0)),
            1.0,
            labeller=LABELLER,
            rng=random.Random(0),
        )
        _match_truth(sab, agents.RandomAgent(random.Random(1)), sab.throws, 3)
        self.assertTrue(sab.throw_games)
        for g in sab.throw_games:
            self.assertIs(type(g), int)


class TestLoadedAgents(unittest.TestCase):
    def test_load_saboteur_and_stealth_start_empty_and_keep_names(self):
        sab = zoo.load("saboteur:1", random.Random(0), LABELLER)
        self.assertEqual((sab.throws, sab.throw_games), ([], []))
        self.assertTrue(sab.name.startswith("saboteur:1:"))
        st = zoo.load("stealth:1:0.5:1", random.Random(0), LABELLER)
        self.assertEqual((st.throws, st.throw_games), ([], []))
        self.assertTrue(st.name.startswith("stealth:1:0.5:1:"))

    def test_load_sandbag_starts_empty_and_keeps_name(self):
        sb = zoo.load("sandbag:slow", random.Random(0), LABELLER)
        self.assertEqual((sb.deviations, sb.deviation_games), ([], []))
        self.assertEqual(sb.name, "sandbag:slow")

    def test_loaded_saboteur_joins_in_a_match(self):
        sab = zoo.load("saboteur:1", random.Random(2), LABELLER)
        opp = agents.RandomAgent(random.Random(2))
        truth = _match_truth(sab, opp, sab.throws, 4)
        self.assertGreaterEqual(len(set(truth)), 2, truth)
        self.assertTrue(_groups_like(sab.throw_games, truth))


class TestWrappedAndEdgeCases(unittest.TestCase):
    def test_noisy_wrapper_forwards_reset_so_records_still_join(self):
        sab = zoo.Saboteur(
            agents.NegamaxAgent(rng=random.Random(7)),
            1.0,
            labeller=LABELLER,
            rng=random.Random(7),
        )
        noisy = agents.NoisyAgent(sab, 0.0, random.Random(7))
        opp = agents.RandomAgent(random.Random(8))
        truth = _match_truth(noisy, opp, sab.throws, 4)
        self.assertGreaterEqual(len(set(truth)), 2, truth)
        self.assertTrue(_groups_like(sab.throw_games, truth))

    def test_games_with_no_entries_still_count(self):
        sab = zoo.Saboteur(
            agents.NegamaxAgent(rng=random.Random(0)),
            1.0,
            labeller=LABELLER,
            rng=random.Random(0),
        )
        board = game.Board.from_moves([0])
        sab.select(board)
        sab.reset(game.Board())
        sab.reset(game.Board())
        sab.select(board)
        sab.reset(game.Board())
        sab.select(board)
        a, b, c = sab.throw_games
        self.assertLess(a, b)
        self.assertLess(b, c)

    def test_sandbagger_fallback_positions_log_nothing(self):
        sb = zoo.Sandbagger(
            LABELLER, "draw", fallback=agents.RandomAgent(random.Random(0))
        )
        big = game.Board(6, 7, 4)
        sb.reset(big)
        sb.select(big)
        self.assertEqual((sb.deviations, sb.deviation_games), ([], []))

    def test_plain_moves_add_no_record(self):
        sab = zoo.Saboteur(
            agents.NegamaxAgent(rng=random.Random(0)),
            0.0,
            labeller=LABELLER,
            rng=random.Random(0),
        )
        _match_truth(sab, agents.RandomAgent(random.Random(0)), sab.throws, 3)
        self.assertEqual((sab.throws, sab.throw_games), ([], []))

    def test_random_select_reset_mix_keeps_lengths_and_grouping(self):
        rng = random.Random(11)
        sab = zoo.Saboteur(
            agents.NegamaxAgent(rng=random.Random(11)),
            0.7,
            labeller=LABELLER,
            rng=random.Random(11),
        )
        sb = zoo.Sandbagger(LABELLER, "slow", rng=random.Random(11))
        boards = [game.Board.from_moves(m) for m in ([], [0], [4], [0, 4])]
        g = 0
        sab_truth, sb_truth = [], []
        for _ in range(200):
            if rng.random() < 0.2:
                sab.reset(game.Board())
                sb.reset(game.Board())
                g += 1
                continue
            board = rng.choice(boards)
            n, m = len(sab.throws), len(sb.deviations)
            sab.select(board)
            sb.select(board)
            sab_truth.extend([g] * (len(sab.throws) - n))
            sb_truth.extend([g] * (len(sb.deviations) - m))
            self.assertEqual(len(sab.throws), len(sab.throw_games))
            self.assertEqual(len(sb.deviations), len(sb.deviation_games))
        self.assertGreaterEqual(len(set(sab_truth)), 2)
        self.assertTrue(_groups_like(sab.throw_games, sab_truth))
        self.assertTrue(_groups_like(sb.deviation_games, sb_truth))


if __name__ == "__main__":
    unittest.main()
