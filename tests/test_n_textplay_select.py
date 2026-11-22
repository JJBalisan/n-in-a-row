"""Selector checks for #40, written before reading any solver diff."""

import random
import unittest

from n_in_a_row import agents, arena, game
from n_in_a_row.harness import textplay


class _Recorder(agents.Agent):
    name = "recorder"

    def __init__(self):
        self.resets = []
        self.reset_boards = []
        self.observed = []
        self.observed_boards = []
        self.selects = 0

    def reset(self, board):
        self.resets.append(board.ply)
        self.reset_boards.append(board)

    def observe(self, board, move):
        self.observed.append((board.ply, move))
        self.observed_boards.append(board)

    def select(self, board):
        self.selects += 1
        return board.legal_moves()[0]


class _CountingModel:
    def __init__(self, reply="no idea"):
        self.reply = reply
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self.reply


def _expected(moves):
    return [(i + 1, m) for i, m in enumerate(moves)]


class TextAgentSelectTest(unittest.TestCase):
    def test_hooks_never_call_the_model(self):
        model = _CountingModel()
        text = textplay.TextAgent(model, fallback=_Recorder())
        board = game.Board(3)
        text.reset(board)
        board.push(4)
        text.observe(board, 4)
        self.assertEqual(model.calls, 0)
        self.assertEqual(text.turns, [])

    def test_reset_keeps_own_counters(self):
        # The verifier asked solvers to leave TextAgent's own state alone.
        text = textplay.TextAgent(lambda p: "no idea", fallback=_Recorder())
        arena.play_game(text, agents.RandomAgent(random.Random(2)))
        turns = list(text.turns)
        failures, fallbacks = text.failures, text.fallbacks
        self.assertGreater(fallbacks, 0)
        text.reset(game.Board(3))
        self.assertEqual(text.turns, turns)
        self.assertEqual(text.failures, failures)
        self.assertEqual(text.fallbacks, fallbacks)

    def test_two_games_reset_each_time(self):
        rec = _Recorder()
        text = textplay.TextAgent(lambda p: "no idea", fallback=rec)
        first = arena.play_game(text, agents.RandomAgent(random.Random(4)))
        second = arena.play_game(text, agents.RandomAgent(random.Random(5)))
        self.assertEqual(rec.resets, [0, 0])
        self.assertEqual(
            rec.observed, _expected(first.moves) + _expected(second.moves)
        )

    def test_playing_second_fallback_hears_opponent_first(self):
        rec = _Recorder()
        text = textplay.TextAgent(lambda p: "no idea", fallback=rec)
        record = arena.play_game(agents.RandomAgent(random.Random(6)), text)
        self.assertEqual(rec.resets, [0])
        self.assertEqual(rec.observed, _expected(record.moves))
        self.assertEqual(text.fallbacks, rec.selects)

    def test_forwards_the_same_board_object(self):
        rec = _Recorder()
        text = textplay.TextAgent(lambda p: "no idea", fallback=rec)
        board = game.Board(3)
        text.reset(board)
        board.push(0)
        text.observe(board, 0)
        self.assertIs(rec.reset_boards[0], board)
        self.assertIs(rec.observed_boards[0], board)

    def test_nested_text_agent_forwards_down_the_chain(self):
        rec = _Recorder()
        inner = textplay.TextAgent(lambda p: "no idea", fallback=rec)
        outer = textplay.TextAgent(lambda p: "no idea", fallback=inner)
        record = arena.play_game(outer, agents.RandomAgent(random.Random(7)))
        self.assertEqual(rec.resets, [0])
        self.assertEqual(rec.observed, _expected(record.moves))

    def test_hooks_return_none(self):
        text = textplay.TextAgent(lambda p: "MOVE: a1", fallback=_Recorder())
        board = game.Board(3)
        self.assertIsNone(text.reset(board))
        board.push(4)
        self.assertIsNone(text.observe(board, 4))


if __name__ == "__main__":
    unittest.main()
