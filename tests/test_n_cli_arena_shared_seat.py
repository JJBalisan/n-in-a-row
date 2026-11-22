"""#44: one agent object in both seats gets each hook once, not twice."""

import unittest

from n_in_a_row import agents, arena


class CountingAgent(agents.Agent):
    """Plays the first legal move and counts its hook calls."""

    name = "counting"

    def __init__(self):
        self.resets = 0
        self.observes = 0
        self.selects = 0

    def reset(self, board):
        self.resets += 1

    def observe(self, board, move):
        self.observes += 1

    def select(self, board):
        self.selects += 1
        return board.legal_moves()[0]


class TestSharedSeat(unittest.TestCase):
    def test_one_object_both_seats_play_game(self):
        a = CountingAgent()
        rec = arena.play_game(a, a)
        plies = len(rec.moves)
        self.assertEqual(a.selects, plies)
        self.assertEqual(a.resets, 1, "reset called more than once per game")
        self.assertEqual(
            a.observes, plies, "observe called more than once per move"
        )

    def test_one_object_both_seats_play_match(self):
        a = CountingAgent()
        records = []
        arena.play_match(a, a, 3, records=records)
        plies = sum(len(r.moves) for r in records)
        self.assertEqual(a.resets, 3)
        self.assertEqual(a.observes, plies)

    def test_two_equal_objects_each_get_every_hook(self):
        # Guards a fix that dedupes by name or equality, not identity.
        a, b = CountingAgent(), CountingAgent()
        rec = arena.play_game(a, b)
        plies = len(rec.moves)
        for agent in (a, b):
            self.assertEqual(agent.resets, 1)
            self.assertEqual(agent.observes, plies)


if __name__ == "__main__":
    unittest.main()
