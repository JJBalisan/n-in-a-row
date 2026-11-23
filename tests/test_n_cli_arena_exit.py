"""PR #78 fix round 2: only a human seat's own quit becomes exit 0.

An agent that calls ``sys.exit`` itself must exit exactly as on main
(v1.1.4): ``sys.exit("boom")`` exits 1 with "boom" on stderr,
``sys.exit()`` exits 0 with nothing on stderr, ``sys.exit(3)`` exits 3.
stdin is never read: every run gets an explicit, empty or scripted stdin.
"""

import contextlib
import io
import os
import subprocess
import sys
import unittest
from unittest import mock

from n_in_a_row import agents, cli

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)
_MODULE = os.path.splitext(os.path.basename(__file__))[0]


class _ExitAgent(agents.Agent):
    exit_args = ()

    def select(self, board):
        sys.exit(*self.exit_args)


class StrExit(_ExitAgent):
    exit_args = ("boom",)


class BareExit(_ExitAgent):
    exit_args = ()


class IntExit(_ExitAgent):
    exit_args = (3,)


def _argvs(spec):
    return {
        "match": ["match", "--a", spec, "--b", "random", "--games", "1"],
        # The agent moves first as X, so it exits before any prompt.
        "play": ["play", "--opponent", spec, "--side", "O"],
    }


def _run(argv, stdin_text=""):
    """Runs cli.main in-process; returns (code, SystemExit or None, err)."""
    out = io.StringIO()
    err = io.StringIO()
    with (
        mock.patch("sys.stdin", io.StringIO(stdin_text)),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        try:
            return cli.main(argv), None, err.getvalue()
        except SystemExit as exc:
            return None, exc, err.getvalue()


class TestAgentExitPassesThrough(unittest.TestCase):
    """main must not catch an agent's own SystemExit, as on main."""

    def _check(self, cls, want_code):
        spec = f"{__name__}:{cls.__name__}"
        for cmd, argv in _argvs(spec).items():
            with self.subTest(cmd=cmd, agent=cls.__name__):
                code, exc, err = _run(argv)
                self.assertIsNone(
                    code, f"main returned {code!r} instead of re-raising"
                )
                self.assertIsNotNone(exc)
                self.assertIs(type(exc), SystemExit)
                self.assertEqual(exc.code, want_code)
                self.assertEqual(err, "", "main printed the exit itself")

    def test_string_exit(self):
        self._check(StrExit, "boom")

    def test_bare_exit(self):
        self._check(BareExit, None)

    def test_int_exit(self):
        self._check(IntExit, 3)


class TestAgentExitProcess(unittest.TestCase):
    """The same, end to end: exit code and stderr of the real process."""

    def _proc(self, argv):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (_TESTS, env.get("PYTHONPATH")) if p
        )
        return subprocess.run(  # noqa: S603 (our own interpreter)
            [sys.executable, "-m", "n_in_a_row", *argv],
            cwd=_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

    def test_exit_codes_and_stderr_match_main(self):
        cases = [
            (StrExit, 1, "boom\n"),
            (BareExit, 0, ""),
            (IntExit, 3, ""),
        ]
        for cls, want_rc, want_err in cases:
            spec = f"{_MODULE}:{cls.__name__}"
            for cmd, argv in _argvs(spec).items():
                with self.subTest(cmd=cmd, agent=cls.__name__):
                    proc = self._proc(argv)
                    self.assertEqual(proc.returncode, want_rc, proc.stderr)
                    self.assertEqual(proc.stderr, want_err)


class TestHumanQuitStillZero(unittest.TestCase):
    """A human seat's own quit in a match still ends with exit 0."""

    def test_quit_as_a_returns_zero(self):
        argv = ["match", "--a", "human", "--b", "random", "--games", "1"]
        code, exc, err = _run(argv, "quit\n")
        self.assertIsNone(exc, f"SystemExit escaped main: {exc!r}")
        self.assertEqual(code, 0)
        self.assertIn("resigned", err)

    def test_input_ends_as_b_returns_zero(self):
        argv = [
            "match",
            "--a",
            "random",
            "--b",
            "human",
            "--games",
            "1",
            "--no-alternate",
        ]
        code, exc, err = _run(argv, "")
        self.assertIsNone(exc, f"SystemExit escaped main: {exc!r}")
        self.assertEqual(code, 0)
        self.assertIn("stopped", err)


if __name__ == "__main__":
    unittest.main()
