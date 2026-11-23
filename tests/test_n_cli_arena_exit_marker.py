"""PR #78 fix round 3: a human quit is a marker HumanAgent raises itself.

Only :class:`agents.HumanQuit`, raised at HumanAgent's own quit sites
(the quit command, end of input), becomes exit 0. Any other SystemExit,
even one raised inside a HumanAgent (a subclass's own code, or its
``input_fn``), exits as on v1.1.4 (cb3e87a), code and stderr alike.
stdin is never read: every run gets an explicit stdin.
"""

import contextlib
import io
import os
import subprocess
import sys
import unittest
from unittest import mock

from n_in_a_row import agents, arena, cli, game

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)
_MODULE = os.path.splitext(os.path.basename(__file__))[0]


class HumanSub(agents.HumanAgent):
    def select(self, board):
        sys.exit(3)


def _exit_int(prompt):
    sys.exit(3)


def _exit_bare(prompt):
    sys.exit()


def int_input_human():
    return agents.HumanAgent(input_fn=_exit_int)


def bare_input_human():
    return agents.HumanAgent(input_fn=_exit_bare)


def _argvs(spec):
    return {
        "match": ["match", "--a", spec, "--b", "random", "--games", "1"],
        # The agent moves first as X, so it exits before any other prompt.
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


class TestMarker(unittest.TestCase):
    def test_marker_is_a_system_exit_and_kept_in_arena(self):
        self.assertTrue(issubclass(agents.HumanQuit, SystemExit))
        self.assertIs(arena.HumanQuit, agents.HumanQuit)

    def test_quit_command_raises_marker(self):
        human = agents.HumanAgent(input_fn=lambda p: "quit")
        with self.assertRaises(agents.HumanQuit) as ctx:
            human.select(game.Board(3))
        self.assertEqual(ctx.exception.code, "resigned")

    def test_end_of_input_raises_marker(self):
        def eof(prompt):
            raise EOFError

        human = agents.HumanAgent(input_fn=eof)
        with self.assertRaises(agents.HumanQuit) as ctx:
            human.select(game.Board(3))
        self.assertEqual(ctx.exception.code, "\nstopped")

    def test_input_fn_exit_is_not_marked(self):
        human = int_input_human()
        with self.assertRaises(SystemExit) as ctx:
            human.select(game.Board(3))
        self.assertIs(type(ctx.exception), SystemExit)
        self.assertEqual(ctx.exception.code, 3)


class TestHumanSeatOwnExitPassesThrough(unittest.TestCase):
    def _check(self, name, want_code):
        for cmd, argv in _argvs(f"{__name__}:{name}").items():
            with self.subTest(cmd=cmd, agent=name):
                code, exc, err = _run(argv)
                self.assertIsNone(
                    code, f"main returned {code!r} instead of re-raising"
                )
                self.assertIsNotNone(exc)
                self.assertNotIsInstance(exc, agents.HumanQuit)
                self.assertEqual(exc.code, want_code)
                self.assertEqual(err, "", "main printed the exit itself")

    def test_subclass_select_exit_3(self):
        self._check("HumanSub", 3)

    def test_input_fn_exit_int(self):
        self._check("int_input_human", 3)

    def test_input_fn_bare_exit(self):
        self._check("bare_input_human", None)


class TestHumanSeatOwnExitProcess(unittest.TestCase):
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

    def test_exit_codes_and_stderr_match_v114(self):
        cases = [
            ("HumanSub", 3, ""),
            ("int_input_human", 3, ""),
            ("bare_input_human", 0, ""),
        ]
        for name, want_rc, want_err in cases:
            for cmd, argv in _argvs(f"{_MODULE}:{name}").items():
                with self.subTest(cmd=cmd, agent=name):
                    proc = self._proc(argv)
                    self.assertEqual(proc.returncode, want_rc, proc.stderr)
                    self.assertEqual(proc.stderr, want_err)


class TestHumanQuitIsClean(unittest.TestCase):
    def test_builtin_human_quit_in_match(self):
        argv = ["match", "--a", "human", "--b", "random", "--games", "1"]
        code, exc, err = _run(argv, "quit\n")
        self.assertIsNone(exc, f"SystemExit escaped main: {exc!r}")
        self.assertEqual(code, 0)
        self.assertIn("resigned", err)

    def test_noisy_wrapped_human_quit_in_match(self):
        argv = [
            "match",
            "--a",
            "noisy:0:human",
            "--b",
            "random",
            "--games",
            "1",
        ]
        code, exc, err = _run(argv, "quit\n")
        self.assertIsNone(exc, f"SystemExit escaped main: {exc!r}")
        self.assertEqual(code, 0)
        self.assertIn("resigned", err)

    def test_noisy_wrapped_human_quit_in_play(self):
        argv = ["play", "--opponent", "noisy:0:human", "--side", "O"]
        code, exc, _ = _run(argv, "quit\n")
        self.assertIsNone(exc, f"SystemExit escaped main: {exc!r}")
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
