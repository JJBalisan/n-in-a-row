"""Selector's adversarial checks for #38, #44, #49, #53 and #57.

Written from the issue text and the verifier's notes, before reading any
solver diff. Counts, identities and outputs only; no wall-clock timing.
"""

import contextlib
import io
import os
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

from n_in_a_row import agents, arena, cli, game

_MOD = "_n_cli_arena_select_fake"


class _Counting(agents.Agent):
    """First legal move; counts hook calls. No rng attribute."""

    name = "counting"

    def __init__(self):
        self.resets = 0
        self.observes = 0

    def select(self, board):
        return board.legal_moves()[0]

    def reset(self, board):
        self.resets += 1

    def observe(self, board, move):
        self.observes += 1


class _Unhashable(_Counting):
    """Compares by value and so has no hash."""

    def __eq__(self, other):
        return isinstance(other, _Unhashable)

    __hash__ = None


class _AlwaysEqual(_Counting):
    """Equal to every other _AlwaysEqual, with one shared hash."""

    def __eq__(self, other):
        return isinstance(other, _AlwaysEqual)

    def __hash__(self):
        return 1


def _install_fake_module():
    mod = types.ModuleType(_MOD)
    mod.AGENT = _Counting()
    mod.Factory = _Counting
    sys.modules[_MOD] = mod
    return mod


def _run(argv, stdin_text):
    """Runs cli.main with piped stdin; returns (code, escaped exc, stdout)."""
    out = io.StringIO()
    with (
        mock.patch("sys.stdin", io.StringIO(stdin_text)),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        try:
            return cli.main(argv), None, out.getvalue()
        except (agents.Undo, SystemExit) as exc:
            return None, exc, out.getvalue()


# ------------------------------------------------------------------ #38


class TestHumanSeats(unittest.TestCase):
    def _no_escape(self, argv, stdin_text):
        code, exc, out = _run(argv, stdin_text)
        if exc is not None:
            self.fail(f"{type(exc).__name__} escaped main: {exc!r} {argv}")
        self.assertIsInstance(code, int)
        return code, out

    def test_argparse_errors_still_exit_2(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as cm,
        ):
            cli.main(["no-such-command"])
        self.assertEqual(cm.exception.code, 2)

    def test_play_human_opponent_moves_first_undo(self):
        # You are O; the human opponent is X and types undo at once.
        code, _ = self._no_escape(
            ["play", "--opponent", "human", "--side", "O"], "undo\nquit\n"
        )
        self.assertEqual(code, 0)

    def test_play_human_opponent_eof(self):
        code, _ = self._no_escape(["play", "--opponent", "human"], "a1\n")
        self.assertEqual(code, 0)

    def test_play_human_opponent_full_game(self):
        # X: a1 b1 c1, O: a2 b2. Somebody wins; the game ends normally.
        code, out = self._no_escape(
            ["play", "--opponent", "human"], "a1\na2\nb1\nb2\nc1\n"
        )
        self.assertEqual(code, 0)
        self.assertIn("You win", out)

    def test_match_human_both_seats_undo(self):
        self._no_escape(
            ["match", "--a", "human", "--b", "human", "--games", "1"],
            "undo\nquit\n",
        )

    def test_match_human_eof(self):
        self._no_escape(
            ["match", "--a", "human", "--b", "random", "--games", "1"], ""
        )

    def test_match_human_both_seats_full_game(self):
        code, out = self._no_escape(
            ["match", "--a", "human", "--b", "human", "--games", "1"],
            "a1\na2\nb1\nb2\nc1\n",
        )
        self.assertEqual(code, 0)
        self.assertIn("over 1 games: 1-0-0", out)

    def test_keyboard_interrupt_still_130(self):
        with mock.patch.object(
            arena, "play_match", side_effect=KeyboardInterrupt
        ):
            code, exc, _ = _run(
                ["match", "--a", "random", "--b", "random", "--games", "1"],
                "",
            )
        self.assertIsNone(exc)
        self.assertEqual(code, 130)


# ------------------------------------------------------------------ #44


class TestSharedSeat(unittest.TestCase):
    def test_unhashable_agent_in_both_seats(self):
        a = _Unhashable()
        rec = arena.play_game(a, a)
        self.assertEqual(a.resets, 1)
        self.assertEqual(a.observes, len(rec.moves))

    def test_equal_but_distinct_agents_each_get_hooks(self):
        x, o = _AlwaysEqual(), _AlwaysEqual()
        rec = arena.play_game(x, o)
        for seat in (x, o):
            self.assertEqual(seat.resets, 1)
            self.assertEqual(seat.observes, len(rec.moves))

    def test_unhashable_distinct_agents_each_get_hooks(self):
        x, o = _Unhashable(), _Unhashable()
        rec = arena.play_game(x, o)
        for seat in (x, o):
            self.assertEqual(seat.resets, 1)
            self.assertEqual(seat.observes, len(rec.moves))

    def test_on_move_once_per_move(self):
        a = _Counting()
        calls = []
        rec = arena.play_game(a, a, on_move=lambda b, m, mv: calls.append(mv))
        self.assertEqual(calls, rec.moves)

    def test_play_match_same_object(self):
        a = _Counting()
        records = []
        arena.play_match(a, a, 4, records=records)
        self.assertEqual(len(records), 4)
        self.assertEqual(a.resets, 4)
        self.assertEqual(a.observes, sum(len(r.moves) for r in records))

    def test_cli_match_module_agent_both_seats(self):
        mod = _install_fake_module()
        self.addCleanup(sys.modules.pop, _MOD, None)
        code, exc, _ = _run(
            [
                "match",
                "--a",
                f"{_MOD}:AGENT",
                "--b",
                f"{_MOD}:AGENT",
                "--games",
                "3",
                "--quiet",
            ],
            "",
        )
        self.assertIsNone(exc)
        self.assertEqual(code, 0)
        self.assertEqual(mod.AGENT.resets, 3)


# ------------------------------------------------------------------ #49


class TestFromJsonEdges(unittest.TestCase):
    def test_round_trip_edges(self):
        for rec in (
            arena.GameRecord(3, 3, 3),
            arena.GameRecord(3, 3, 3, [4, 0]),
            arena.GameRecord(3, 3, 3, [0, 1, 2], game.DRAW),
            arena.GameRecord(4, 5, 3, [0, 5, 1, 6, 2], game.X),
        ):
            with self.subTest(rec=rec):
                back = arena.GameRecord.from_json(rec.to_json())
                self.assertEqual(back, rec)
                self.assertIsInstance(back.moves, list)
                self.assertEqual(back.to_json(), rec.to_json())

    def test_trailing_newline(self):
        rec = arena.GameRecord(3, 3, 3, [4, 0, 8], game.O)
        back = arena.GameRecord.from_json(rec.to_json() + "\n")
        self.assertEqual(back, rec)

    def test_reads_selfplay_out_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sp.jsonl")
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(
                    [
                        "selfplay",
                        "--agent",
                        "random",
                        "--games",
                        "5",
                        "--seed",
                        "2",
                        "--out",
                        path,
                        "--quiet",
                    ]
                )
            with open(path) as fh:
                lines = fh.readlines()
        self.assertEqual(len(lines), 5)
        want = list(arena.selfplay("random", 5, seed=2))
        got = [arena.GameRecord.from_json(ln) for ln in lines]
        self.assertEqual(got, want)


# ------------------------------------------------------------------ #53


_TINY = ("--games", "1", "--depth", "1", "--batch", "1", "--steps", "1")


class TestBenchK(unittest.TestCase):
    def _first(self, extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["bench", *extra, *_TINY])
        return out.getvalue().splitlines()[0]

    def test_matches_board_default(self):
        for extra, rows, cols in (
            (["--rows", "3", "--cols", "5"], 3, 5),
            (["--cols", "4"], 3, 4),
        ):
            with self.subTest(extra=extra):
                k = game.Board(rows, cols, None).k
                self.assertEqual(
                    self._first(extra), f"board {rows}x{cols}, {k} in a row"
                )


# ------------------------------------------------------------------ #57


def _reference(spec, games, rows, cols, k, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        a = agents.load_agent(spec, random.Random(rng.getrandbits(32)))
        b = agents.load_agent(spec, random.Random(rng.getrandbits(32)))
        out.append(arena.play_game(a, b, rows, cols, k))
    return out


class TestSelfplayEdges(unittest.TestCase):
    def test_records_identical_more_specs(self):
        for spec, rows, cols, k in (
            ("perfect", 3, 3, 3),
            ("negamax:2", 4, 4, 3),
            ("noisy:0.5:random", 3, 4, 3),
            ("noisy:0.2:noisy:0.4:greedy", 3, 3, 3),
            ("noisy:0.3:noisy:0.3:perfect", 3, 3, 3),
        ):
            with self.subTest(spec=spec):
                want = _reference(spec, 8, rows, cols, k, 13)
                got = list(arena.selfplay(spec, 8, rows, cols, k, seed=13))
                self.assertEqual(got, want)

    def test_nested_noisy_load_count_constant(self):
        spec = "noisy:0.2:noisy:0.4:greedy"
        real = agents.load_agent
        counts = []
        for games in (2, 6):
            with mock.patch.object(
                agents, "load_agent", side_effect=real
            ) as spy:
                list(arena.selfplay(spec, games, seed=1))
            counts.append(spy.call_count)
        self.assertEqual(counts[0], counts[1])

    def test_interleaved_generators_independent(self):
        want = list(arena.selfplay("noisy:0.3:greedy", 6, seed=5))
        g1 = arena.selfplay("noisy:0.3:greedy", 6, seed=5)
        g2 = arena.selfplay("noisy:0.3:greedy", 6, seed=5)
        got1, got2 = [], []
        for _ in range(6):
            got1.append(next(g1))
            got2.append(next(g2))
        self.assertEqual(got1, want)
        self.assertEqual(got2, want)

    def test_unseeded_and_zero_games(self):
        self.assertEqual(len(list(arena.selfplay("random", 4))), 4)
        self.assertEqual(list(arena.selfplay("random", 0, seed=1)), [])

    def test_module_specs_without_rng(self):
        mod = _install_fake_module()
        self.addCleanup(sys.modules.pop, _MOD, None)
        recs = list(arena.selfplay(f"{_MOD}:Factory", 3, seed=1))
        self.assertEqual(len(recs), 3)
        recs = list(arena.selfplay(f"{_MOD}:AGENT", 3, seed=1))
        self.assertEqual(len(recs), 3)
        # One module-level object in both seats: one reset per game.
        self.assertEqual(mod.AGENT.resets, 3)


# ------------------------------------------------- differential (diffs)


class _NoSeedGenerator:
    """Stands in for numpy.random.Generator: random(), but no seed()."""

    def __init__(self):
        self._r = random.Random(0)

    def random(self):
        return self._r.random()


class _OwnGenerator(_Counting):
    """A module agent that keeps its own, non-stdlib generator as .rng."""

    def __init__(self):
        super().__init__()
        self.rng = _NoSeedGenerator()


class _Exits(_Counting):
    """An agent that ends the program with its own exit code."""

    def select(self, board):
        raise SystemExit(3)


class TestDifferential(unittest.TestCase):
    def test_selfplay_leaves_module_agent_rng_alone(self):
        mod = _install_fake_module()
        mod.OwnGen = _OwnGenerator
        self.addCleanup(sys.modules.pop, _MOD, None)
        recs = list(arena.selfplay(f"{_MOD}:OwnGen", 3, seed=1))
        self.assertEqual(len(recs), 3)

    def test_agent_exit_code_still_propagates(self):
        # Not a human seat: an agent's own integer exit code is kept, as
        # before the fix; only the human's quit and undo are caught.
        mod = _install_fake_module()
        mod.Exits = _Exits
        self.addCleanup(sys.modules.pop, _MOD, None)
        code, exc, _ = _run(
            ["match", "--a", f"{_MOD}:Exits", "--b", "random", "--games", "1"],
            "",
        )
        self.assertIsInstance(exc, SystemExit, f"main returned {code}")
        self.assertEqual(exc.code, 3)


if __name__ == "__main__":
    unittest.main()
