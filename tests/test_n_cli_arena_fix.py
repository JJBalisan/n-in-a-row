"""Fresh-check fix: play keeps an opponent agent's own exit code.

Before #38's fix, cmd_play caught SystemExit only around the human's
select, so an opponent agent that called sys.exit(3) ended the program
with code 3. The human's quit still returns 0.
"""

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

from n_in_a_row import agents, cli

_MOD = "_n_cli_arena_fix_fake"


class _Exits(agents.Agent):
    """An opponent that ends the program with its own exit code."""

    name = "exits"

    def select(self, board):
        raise SystemExit(3)


class _Undoes(agents.Agent):
    """An opponent that raises Undo, which only a human seat may mean."""

    name = "undoes"

    def __init__(self):
        self.calls = 0

    def select(self, board):
        # Undo once, then exit, so a wrong catch cannot loop forever.
        self.calls += 1
        if self.calls == 1:
            raise agents.Undo()
        raise SystemExit(5)


def _run(argv, stdin_text):
    """Runs cli.main with piped stdin; returns (code, escaped exc)."""
    with (
        mock.patch("sys.stdin", io.StringIO(stdin_text)),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        try:
            return cli.main(argv), None
        except SystemExit as exc:
            return None, exc


class TestPlayOpponentExit(unittest.TestCase):
    def setUp(self):
        mod = types.ModuleType(_MOD)
        mod.Exits = _Exits
        mod.Undoes = _Undoes
        sys.modules[_MOD] = mod
        self.addCleanup(sys.modules.pop, _MOD, None)

    def test_opponent_exit_code_propagates(self):
        code, exc = _run(
            ["play", "--opponent", f"{_MOD}:Exits", "--side", "O"], ""
        )
        self.assertIsInstance(exc, SystemExit, f"main returned {code}")
        self.assertEqual(exc.code, 3)

    def test_opponent_undo_is_not_a_takeback(self):
        # The engine raising Undo must not loop as if a human took back;
        # it leaves cmd_play (main reports it as an error, code 2).
        code, exc = _run(
            ["play", "--opponent", f"{_MOD}:Undoes", "--side", "O"], ""
        )
        self.assertIsNone(exc)
        self.assertEqual(code, 2)

    def test_human_quit_still_returns_zero(self):
        code, exc = _run(["play", "--opponent", "random"], "quit\n")
        self.assertIsNone(exc)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
