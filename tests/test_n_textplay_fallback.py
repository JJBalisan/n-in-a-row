"""Checks for #40: TextAgent forwards reset and observe to its fallback."""

import random
import unittest

from n_in_a_row import agents, arena, game
from n_in_a_row.harness import textplay


class _Recorder(agents.Agent):
    """Plays the first legal cell and records every hook call."""

    name = "recorder"

    def __init__(self):
        self.resets = []
        self.observed = []
        self.selects = 0

    def reset(self, board):
        self.resets.append(board.ply)

    def observe(self, board, move):
        self.observed.append((board.ply, move))

    def select(self, board):
        self.selects += 1
        return board.legal_moves()[0]


def _expected_observes(moves):
    return [(i + 1, m) for i, m in enumerate(moves)]


class TextAgentFallbackHooksTest(unittest.TestCase):
    def test_fallback_hears_reset_and_every_move(self):
        rec = _Recorder()
        text = textplay.TextAgent(lambda p: "no idea", fallback=rec)
        record = arena.play_game(text, agents.RandomAgent(random.Random(1)))
        self.assertEqual(text.fallbacks, rec.selects)
        self.assertGreater(rec.selects, 0)
        self.assertEqual(rec.resets, [0])
        self.assertEqual(rec.observed, _expected_observes(record.moves))

    def test_fallback_hears_moves_the_model_played(self):
        # The model names a legal move on its first turn and fails after,
        # so the fallback must have tracked the moves it did not play.
        replies = iter(["MOVE: a1"])
        rec = _Recorder()
        text = textplay.TextAgent(lambda p: next(replies, "pass"), fallback=rec)
        record = arena.play_game(text, agents.RandomAgent(random.Random(3)))
        self.assertEqual(record.moves[0], 0)
        self.assertEqual(rec.resets, [0])
        self.assertEqual(rec.observed, _expected_observes(record.moves))
        self.assertEqual(text.fallbacks, rec.selects)

    def test_reset_and_observe_forward_the_arguments(self):
        rec = _Recorder()
        text = textplay.TextAgent(lambda p: "MOVE: a1", fallback=rec)
        board = game.Board(3)
        text.reset(board)
        board.push(4)
        text.observe(board, 4)
        self.assertEqual(rec.resets, [0])
        self.assertEqual(rec.observed, [(1, 4)])
        self.assertEqual(rec.selects, 0)

    def test_no_fallback_hooks_are_safe(self):
        text = textplay.TextAgent(lambda p: "MOVE: a1")
        board = game.Board(3)
        text.reset(board)
        board.push(4)
        text.observe(board, 4)
        self.assertEqual(text.select(board), 0)


if __name__ == "__main__":
    unittest.main()
