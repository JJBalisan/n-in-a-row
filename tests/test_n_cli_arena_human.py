"""#38: `human` as --opponent, --a or --b must not let Undo or SystemExit
escape `cli.main`.

These checks pin only what the issue reports. What "undo" does inside a
match, the wording printed, and the exit code of a quit match are left to
the fix (see found/38-decision.md in the sweep lane folder).
"""

import contextlib
import io
import unittest
from unittest import mock

from n_in_a_row import agents, cli


def _run(argv, stdin_text):
    """Runs cli.main with piped stdin; returns (result, escaped exception)."""
    out = io.StringIO()
    err = io.StringIO()
    with (
        mock.patch("sys.stdin", io.StringIO(stdin_text)),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        try:
            return cli.main(argv), None
        except (agents.Undo, SystemExit) as exc:
            return None, exc


class TestPlayHumanOpponent(unittest.TestCase):
    def test_undo_at_opponent_prompt_does_not_escape(self):
        # X (you) plays a1, the human opponent types undo, then quit.
        code, exc = _run(["play", "--opponent", "human"], "a1\nundo\nquit\n")
        if exc is not None:
            self.fail(
                f"{type(exc).__name__} escaped main: {exc!r} "
                "(play --opponent human, opponent typed undo)"
            )
        self.assertEqual(code, 0)

    def test_quit_at_opponent_prompt_returns_zero(self):
        # cmd_play's docstring: 0 when the game ends or the player quits.
        code, exc = _run(["play", "--opponent", "human"], "a1\nquit\n")
        if exc is not None:
            self.fail(
                f"{type(exc).__name__} escaped main: {exc!r} "
                "(play --opponent human, opponent typed quit)"
            )
        self.assertEqual(code, 0)


class TestMatchHumanSide(unittest.TestCase):
    def _assert_no_escape(self, argv, stdin_text):
        code, exc = _run(argv, stdin_text)
        if exc is not None:
            self.fail(
                f"{type(exc).__name__} escaped main: {exc!r} "
                f"({' '.join(argv)!r}, stdin {stdin_text!r})"
            )
        self.assertIsInstance(code, int)

    def test_undo_as_a_does_not_escape(self):
        self._assert_no_escape(
            [
                "match",
                "--a",
                "human",
                "--b",
                "random",
                "--games",
                "1",
                "--seed",
                "1",
            ],
            "undo\nquit\n",
        )

    def test_quit_as_a_does_not_escape(self):
        self._assert_no_escape(
            [
                "match",
                "--a",
                "human",
                "--b",
                "random",
                "--games",
                "1",
                "--seed",
                "1",
            ],
            "quit\n",
        )

    def test_undo_as_b_does_not_escape(self):
        self._assert_no_escape(
            [
                "match",
                "--a",
                "random",
                "--b",
                "human",
                "--games",
                "1",
                "--seed",
                "1",
                "--no-alternate",
            ],
            "undo\nquit\n",
        )


if __name__ == "__main__":
    unittest.main()
