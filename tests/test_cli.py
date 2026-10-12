"""Tests for the command line, driven in-process through cli.main(argv).

Stdlib only:  python3 -m unittest discover -s tests
"""

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))
sys.path.insert(0, _TESTS)  # so agent specs can name fake_agent

from tictactoe import cli
from tictactoe.game import DRAW, Board, O, X


def run_cli(argv):
    """Runs cli.main with captured output and colour forced off.

    Args:
        argv: The argument list, without the program name.

    Returns:
        A (exit_code, stdout, stderr) tuple. An argparse rejection, which
        raises SystemExit, is reported through its exit code.
    """
    out, err = io.StringIO(), io.StringIO()
    with (
        mock.patch.object(cli, "_color", return_value=False),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def parse_solve(stdout):
    """Splits solve output into its headline, best move and per-move table.

    Args:
        stdout: Everything cmd_solve printed.

    Returns:
        A (side, outcome, best_label, table) tuple where table maps each
        move label to its outcome string.
    """
    lines = stdout.splitlines()
    head_i = next(i for i, ln in enumerate(lines) if " to move: " in ln)
    side, outcome = lines[head_i].split(" to move: ", 1)
    best_line = lines[head_i + 1].split()
    assert best_line[0] == "best", lines[head_i + 1]
    table = {}
    for ln in lines[head_i + 2 :]:
        if ln.strip():
            label, result = ln.strip().split(None, 1)
            table[label] = result
    return side, outcome.strip(), best_line[1], table


def read_jsonl(path):
    """Reads one JSON object per line.

    Args:
        path: The file to read.

    Returns:
        A list of the decoded objects.
    """
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def summary_counts(stdout):
    """Pulls W-D-L and the per-colour splits out of a match summary.

    Args:
        stdout: Everything cmd_match printed.

    Returns:
        A dict with 'games', 'wdl', 'as_x' and 'as_o' entries.
    """
    m = re.search(r"over (\d+) games: (\d+)-(\d+)-(\d+) \(W-D-L\)", stdout)
    x = re.search(r"as X: (\d+)-(\d+)-(\d+)", stdout)
    o = re.search(r"as O: (\d+)-(\d+)-(\d+)", stdout)
    return {
        "games": int(m.group(1)),
        "wdl": tuple(int(g) for g in m.groups()[1:]),
        "as_x": tuple(int(g) for g in x.groups()),
        "as_o": tuple(int(g) for g in o.groups()),
    }


def replay_record(rec):
    """Replays a JSONL game record on a fresh board.

    Args:
        rec: One decoded record from --out.

    Returns:
        The board after every recorded move.
    """
    board = Board(rec["rows"], rec["cols"], rec["k"])
    for move in rec["moves"]:
        board.push(move)
    return board


class TestShape(unittest.TestCase):
    """--size, --rows, --cols and -k resolve into (rows, cols, k)."""

    def test_precedence_table(self):
        parser = cli.build_parser()
        cases = [
            ([], (3, 3, None)),
            (["--size", "5"], (5, 5, None)),
            (["--rows", "2"], (2, 2, None)),  # cols follows rows
            (["--cols", "6"], (3, 6, None)),  # rows falls back to size
            (["--size", "5", "--cols", "6"], (5, 6, None)),
            (["--size", "5", "--rows", "2"], (2, 2, None)),
            (["--rows", "2", "--cols", "7"], (2, 7, None)),
            (["--size", "9", "--rows", "2", "--cols", "7"], (2, 7, None)),
            (["--size", "4", "-k", "3"], (4, 4, 3)),
            (["--rows", "6", "--cols", "7", "--k", "4"], (6, 7, 4)),
        ]
        for command in ("solve", "bench", "selfplay", "play"):
            for flags, expected in cases:
                with self.subTest(command=command, flags=flags):
                    args = parser.parse_args([command, *flags])
                    self.assertEqual(cli._shape(args), expected)

    def test_match_shape_reaches_the_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "games.jsonl")
            code, _, _ = run_cli(
                [
                    "match",
                    "--a",
                    "random",
                    "--b",
                    "random",
                    "--games",
                    "2",
                    "--seed",
                    "1",
                    "--size",
                    "9",
                    "--rows",
                    "2",
                    "--cols",
                    "5",
                    "-k",
                    "3",
                    "--out",
                    out,
                ]
            )
            self.assertEqual(code, 0)
            for rec in read_jsonl(out):
                self.assertEqual(
                    (rec["rows"], rec["cols"], rec["k"]), (2, 5, 3)
                )


class TestSolve(unittest.TestCase):
    def test_prints_board_headline_and_every_legal_reply(self):
        code, stdout, stderr = run_cli(["solve", "a1", "b2"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("1 X . .", stdout)
        self.assertIn("2 . O .", stdout)
        side, outcome, best, table = parse_solve(stdout)
        self.assertEqual(side, "X")
        self.assertEqual(outcome, "draw")
        self.assertEqual(
            sorted(table), sorted(["b1", "c1", "a2", "c2", "a3", "b3", "c3"])
        )
        self.assertTrue(all(v == "draw" for v in table.values()), table)
        self.assertIn(best, table)

    def test_headline_agrees_with_its_table(self):
        """The headline outcome is the table's row for the best move."""
        positions = [
            ["a1", "b2", "c3", "c1"],  # the audit's reproduction
            ["a1", "b1", "b2"],
            ["b2", "a1", "c3"],
            ["a1", "a2", "b1", "b2"],
            ["a1", "c3", "a3"],
        ]
        for moves in positions:
            with self.subTest(moves=moves):
                code, stdout, _ = run_cli(["solve", *moves])
                self.assertEqual(code, 0)
                _, outcome, best, table = parse_solve(stdout)
                self.assertEqual(outcome, table[best], stdout)
                self.assertNotIn("horizon", stdout)

    def test_mate_distance_in_headline(self):
        code, stdout, _ = run_cli(["solve", "a1", "b2", "c3", "c1"])
        self.assertEqual(code, 0)
        side, outcome, best, table = parse_solve(stdout)
        self.assertEqual((side, outcome, best), ("X", "win in 2", "a3"))
        self.assertEqual(table["a3"], "win in 2")

    def test_rectangle_and_k(self):
        code, stdout, _ = run_cli(
            ["solve", "--rows", "2", "--cols", "4", "-k", "2", "a1"]
        )
        self.assertEqual(code, 0)
        side, outcome, _, table = parse_solve(stdout)
        self.assertEqual(side, "O")
        self.assertEqual(len(table), 7)
        self.assertTrue(outcome.startswith("loss"), outcome)

    def test_depth_limit_is_flagged(self):
        code, stdout, _ = run_cli(["solve", "--size", "4", "--depth", "1"])
        self.assertEqual(code, 0)
        self.assertIn("(horizon-limited estimate)", stdout)


class TestBadInput(unittest.TestCase):
    """Rejected input exits 2 with a message, not a traceback."""

    def test_bad_moves_exit_2(self):
        for moves, message in [
            (["a1", "a1"], "a1 is already taken"),
            (["z9"], "off the board"),
            (["foo"], "could not read"),
            (["9"], "index 9 is off the board"),
            (["a1", "b1", "a2", "b2", "a3", "c3"], "already over"),
        ]:
            with self.subTest(moves=moves):
                code, stdout, stderr = run_cli(["solve", *moves])
                self.assertEqual(code, 2)
                self.assertTrue(stderr.startswith("error: "), stderr)
                self.assertIn(message, stderr)
                self.assertEqual(stdout, "")

    def test_argparse_rejections_exit_2(self):
        for argv in (
            [],
            ["nonsense"],
            ["solve", "--size", "three"],
            ["play", "--side", "Z"],
            ["match", "--b", "random"],  # --a is required
            ["match", "--a", "random", "--b", "random", "--games", "x"],
        ):
            with self.subTest(argv=argv):
                code, stdout, stderr = run_cli(argv)
                self.assertEqual(code, 2)
                self.assertIn("usage:", stderr)
                self.assertEqual(stdout, "")

    def _assert_clean_exit_2(self, argv):
        try:
            code, _, stderr = run_cli(argv)
        except (ValueError, TypeError, ImportError) as exc:
            self.fail(f"traceback instead of exit 2: {exc!r}")
        self.assertEqual(code, 2)
        self.assertTrue(stderr.startswith("error: "), stderr)

    @unittest.expectedFailure
    def test_unknown_agent_spec_exits_2(self):
        """KNOWN-FAIL: audit P2 'CLI error handling for bad agent specs'."""
        self._assert_clean_exit_2(
            ["match", "--a", "nonsense", "--b", "random", "--games", "1"]
        )

    @unittest.expectedFailure
    def test_missing_agent_module_exits_2(self):
        """KNOWN-FAIL: audit P2 'CLI error handling for bad agent specs'."""
        self._assert_clean_exit_2(
            [
                "match",
                "--a",
                "no_such_module_xyz:agent",
                "--b",
                "random",
                "--games",
                "1",
            ]
        )

    @unittest.expectedFailure
    def test_malformed_numeric_tail_exits_2(self):
        """KNOWN-FAIL: audit P2 '... and malformed numeric tails'."""
        self._assert_clean_exit_2(
            ["match", "--a", "negamax:x", "--b", "random", "--games", "1"]
        )

    @unittest.expectedFailure
    def test_impossible_k_exits_2(self):
        """KNOWN-FAIL (new, not in the audit): Board's ValueError escapes.

        main() catches only InvalidMove, so 'solve --size 3 -k 5' dies with a
        ValueError traceback instead of 'error: ...' and exit code 2.
        """
        self._assert_clean_exit_2(["solve", "--size", "3", "-k", "5"])


class TestMatch(unittest.TestCase):
    def test_summary_and_recorded_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "games.jsonl")
            code, stdout, stderr = run_cli(
                [
                    "match",
                    "--a",
                    "greedy",
                    "--b",
                    "random",
                    "--games",
                    "6",
                    "--seed",
                    "7",
                    "--out",
                    out,
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn("greedy vs random over 6 games", stdout)
            self.assertIn(f"wrote 6 games to {out}", stdout)
            counts = summary_counts(stdout)
            self.assertEqual(sum(counts["wdl"]), 6)
            self.assertEqual(sum(counts["as_x"]), 3)
            self.assertEqual(sum(counts["as_o"]), 3)

            records = read_jsonl(out)
            self.assertEqual(len(records), 6)
            wins = draws = losses = 0
            for i, rec in enumerate(records):
                board = replay_record(rec)
                self.assertTrue(board.is_terminal)
                self.assertEqual(board.winner, rec["winner"])
                a_side = X if i % 2 == 0 else O  # colours alternate
                if rec["winner"] == DRAW:
                    draws += 1
                elif rec["winner"] == a_side:
                    wins += 1
                else:
                    losses += 1
            self.assertEqual((wins, draws, losses), counts["wdl"])

    def test_same_seed_same_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [os.path.join(tmp, f"{n}.jsonl") for n in "ab"]
            for path in paths:
                run_cli(
                    [
                        "match",
                        "--a",
                        "greedy",
                        "--b",
                        "random",
                        "--games",
                        "5",
                        "--seed",
                        "11",
                        "--out",
                        path,
                    ]
                )
            self.assertEqual(read_jsonl(paths[0]), read_jsonl(paths[1]))

    def test_no_alternate_keeps_a_on_x(self):
        code, stdout, _ = run_cli(
            [
                "match",
                "--a",
                "random",
                "--b",
                "random",
                "--games",
                "4",
                "--seed",
                "2",
                "--no-alternate",
            ]
        )
        self.assertEqual(code, 0)
        counts = summary_counts(stdout)
        self.assertEqual(sum(counts["as_x"]), 4)
        self.assertEqual(counts["as_o"], (0, 0, 0))
        self.assertNotIn("wrote", stdout)

    def test_module_attribute_agent(self):
        code, stdout, _ = run_cli(
            [
                "match",
                "--a",
                "fake_agent:FirstLegal",
                "--b",
                "fake_agent:make_agent",
                "--games",
                "2",
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("first-legal vs last-legal over 2 games", stdout)

    def test_ctrl_c_exits_130(self):
        code, _, _ = run_cli(
            [
                "match",
                "--a",
                "fake_agent:Interrupts",
                "--b",
                "random",
                "--games",
                "1",
            ]
        )
        self.assertEqual(code, 130)


class TestSelfplay(unittest.TestCase):
    def test_counts_and_records_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "self.jsonl")
            code, stdout, stderr = run_cli(
                [
                    "selfplay",
                    "--agent",
                    "random",
                    "--games",
                    "12",
                    "--seed",
                    "3",
                    "--size",
                    "4",
                    "-k",
                    "3",
                    "--out",
                    out,
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn("12 games of random", stdout)
            self.assertIn(f"wrote {out}", stdout)
            m = re.search(r"X (\d+)\s+O (\d+)\s+drawn (\d+)", stdout)
            printed = tuple(int(g) for g in m.groups())

            records = read_jsonl(out)
            self.assertEqual(len(records), 12)
            tally = {X: 0, O: 0, DRAW: 0}
            for rec in records:
                self.assertEqual(
                    (rec["rows"], rec["cols"], rec["k"]), (4, 4, 3)
                )
                self.assertEqual(replay_record(rec).winner, rec["winner"])
                tally[rec["winner"]] += 1
            self.assertEqual(printed, (tally[X], tally[O], tally[DRAW]))

    def test_same_seed_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = [
                (os.path.join(tmp, f"{n}.jsonl"), seed)
                for n, seed in (("a", "5"), ("b", "5"), ("c", "6"))
            ]
            paths = [path for path, _ in runs]
            for path, seed in runs:
                run_cli(
                    [
                        "selfplay",
                        "--agent",
                        "noisy:0.5:greedy",
                        "--games",
                        "8",
                        "--seed",
                        seed,
                        "--out",
                        path,
                    ]
                )
            with open(paths[0]) as a, open(paths[1]) as b, open(paths[2]) as c:
                first, second, other = a.read(), b.read(), c.read()
            self.assertEqual(first, second)
            self.assertNotEqual(first, other)

    def test_without_out_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = set(os.listdir(tmp))
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                code, stdout, _ = run_cli(
                    [
                        "selfplay",
                        "--agent",
                        "random",
                        "--games",
                        "3",
                        "--seed",
                        "1",
                    ]
                )
            finally:
                os.chdir(cwd)
            self.assertEqual(code, 0)
            self.assertNotIn("wrote", stdout)
            self.assertEqual(set(os.listdir(tmp)), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
