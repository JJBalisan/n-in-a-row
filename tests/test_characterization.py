"""Golden tests pinning current output, written before the style refactor.

They cover what the other suites leave alone: the interactive play loop,
bench, Board.render and repr, the match summary, the GUI's pure helpers and
argument handling, and the two examples. Each expected value was captured
from the code as it stood, so a failure here means behaviour moved.

Stdlib only:  python3 -m unittest discover -s tests
"""

import contextlib
import io
import os
import re
import sys
import unittest
from unittest import mock

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))
sys.path.insert(0, _TESTS)  # so agent specs can name fake_agent

from n_in_a_row import arena, cli, game, search


def run_cli(argv, stdin=''):
    """Runs cli.main with scripted stdin, captured output and no colour.

    Args:
        argv: The argument list, without the program name.
        stdin: Text fed to input() calls.

    Returns:
        A (exit_code, stdout, stderr) tuple.
    """
    out, err = io.StringIO(), io.StringIO()
    with (
        mock.patch.object(cli, '_color', return_value=False),
        mock.patch.object(sys, 'stdin', io.StringIO(stdin)),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def _require_numpy(test):
    """Skips a NumPy test locally, but fails it when CI is set.

    Args:
        test: The running unittest.TestCase.
    """
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        if os.environ.get('CI'):
            test.fail(
                'numpy is not installed but CI is set; '
                "install the test extra: pip install -e '.[test]'"
            )
        test.skipTest('numpy not installed (set CI=1 to make this a failure)')


EMPTY_3X3 = '   a b c\n1 . . .\n2 . . .\n3 . . .\n\n'

PLAY_WIN = (
    "\n3x3, 3 in a row. You are X against first-legal.\n"
    "Enter a cell like b2. 'undo' takes back, 'quit' leaves.\n\n"
    + EMPTY_3X3
    + 'your move (a1) >    a b c\n1 . . .\n2 . X .\n3 . . .\n\n'
    'first-legal plays a1\n\n'
    '   a b c\n1 O . .\n2 . X .\n3 . . .\n\n'
    'your move (b1) >    a b c\n1 O X .\n2 . X .\n3 . . .\n\n'
    'first-legal plays c1\n\n'
    '   a b c\n1 O X O\n2 . X .\n3 . . .\n\n'
    'your move (a2) >    a b c\n1 O X O\n2 . X .\n3 . X .\n\n'
    'You win on b1-b2-b3.\n'
)

PLAY_LOSS = (
    "\n3x3, 3 in a row. You are O against first-legal.\n"
    "Enter a cell like b2. 'undo' takes back, 'quit' leaves.\n\n"
    + EMPTY_3X3
    + 'first-legal plays a1\n\n'
    '   a b c\n1 X . .\n2 . . .\n3 . . .\n\n'
    'your move (b1) > enter a cell like b2, or 2,3, or a bare index; '
    "'undo' takes back a move, 'quit' leaves\n"
    'your move (b1) >    a b c\n1 X . .\n2 . . .\n3 . . O\n\n'
    'first-legal plays b1\n\n'
    '   a b c\n1 X X .\n2 . . .\n3 . . O\n\n'
    "your move (c1) >   could not read 'zz' as a move\n"
    'your move (c1) >    a b c\n1 X X .\n2 . . .\n3 . O O\n\n'
    'first-legal plays c1\n\n'
    '   a b c\n1 X X X\n2 . . .\n3 . O O\n\n'
    'first-legal wins on a1-b1-c1.\n'
)

PLAY_UNDO_QUIT = (
    "\n3x3, 3 in a row. You are X against perfect.\n"
    "Enter a cell like b2. 'undo' takes back, 'quit' leaves.\n\n"
    + EMPTY_3X3
    + 'your move (a1) >    a b c\n1 . . .\n2 . X .\n3 . . .\n\n'
    'perfect plays a1   [draw, depth 8, 2393 nodes, T]   (T)\n\n'
    '   a b c\n1 O . .\n2 . X .\n3 . . .\n\n'
    'your move (b1) > \n(took back a move)\n\n'
    + EMPTY_3X3
    + 'your move (a1) > \n(nothing to take back)\n\n'
    + EMPTY_3X3
    + 'your move (a1) > resigned\n'
)


class TestPlayTranscripts(unittest.TestCase):
    """The interactive loop, word for word."""

    def test_human_wins(self):
        code, out, err = run_cli(
            ['play', '--opponent', 'fake_agent:FirstLegal'], 'b2\nb1\nb3\n'
        )
        self.assertEqual((code, out, err), (0, PLAY_WIN, ''))

    def test_engine_wins_with_help_and_a_bad_entry(self):
        code, out, err = run_cli(
            ['play', '--opponent', 'fake_agent:FirstLegal', '--side', 'o'],
            '?\nc3\nzz\nb3\n',
        )
        self.assertEqual((code, out, err), (0, PLAY_LOSS, ''))

    def test_verbose_undo_and_quit(self):
        code, out, err = run_cli(
            ['play', '--opponent', 'perfect', '-v'], 'b2\nundo\nu\nq\n'
        )
        out = re.sub(r'\d+\.\d\ds', 'T', out)
        self.assertEqual((code, out, err), (0, PLAY_UNDO_QUIT, ''))

    def test_end_of_input_stops(self):
        code, out, err = run_cli(['play'], '')
        self.assertEqual(code, 0)
        self.assertTrue(out.endswith('your move (a1) > \nstopped\n'), out)
        self.assertEqual(err, '')

    def test_taken_cell_ends_with_exit_2(self):
        # Pins current behaviour: a human move onto a taken cell is not
        # re-prompted; push() raises and main reports it.
        code, _, err = run_cli(
            ['play', '--opponent', 'fake_agent:FirstLegal'], 'b2\na1\n'
        )
        self.assertEqual((code, err), (2, 'error: a1 is already taken\n'))


class TestBench(unittest.TestCase):
    def test_lines(self):
        code, out, err = run_cli(
            [
                'bench',
                '--size',
                '3',
                '-k',
                '3',
                '--games',
                '3',
                '--depth',
                '2',
                '--batch',
                '2',
                '--steps',
                '2',
                '--seed',
                '1',
            ]
        )
        self.assertEqual((code, err), (0, ''))
        out = re.sub(r'\d+\.\d\ds', 'T', out)
        out = re.sub(r'-> \d+ ', '-> R ', out)
        lines = out.splitlines()
        self.assertEqual(
            lines[:4],
            [
                'board 3x3, 3 in a row',
                '  engine:   3 random games, 20 moves in T -> R moves/s',
                '  search:   depth 2 from the empty board, 34 nodes in T '
                '-> R nodes/s',
                '            eval +0.01 (horizon-limited)',
            ],
        )
        self.assertIn(
            lines[4],
            [
                '  batchenv: 2 envs x 2 steps = 4 moves in T -> R moves/s',
                '  batchenv: skipped (numpy not installed)',
            ],
        )
        self.assertEqual(len(lines), 5)


class TestRendering(unittest.TestCase):
    def test_plain_and_colour(self):
        b = game.Board.from_moves([0, 4, 8, 2, 6], 3)
        self.assertEqual(b.render(), '   a b c\n1 X . O\n2 . O .\n3 X . X')
        self.assertEqual(str(b), b.render())
        dim = '\x1b[2m.\x1b[0m'
        self.assertEqual(
            b.render(color=True),
            f'   a b c\n1 X {dim} O\n2 {dim} O {dim}\n'
            f'3 \x1b[1;36mX\x1b[0m {dim} X',
        )
        self.assertEqual(repr(b), '<Board 3x3 k=3 ply=5 to_move=O>')

    def test_win_line_and_wide_board(self):
        b = game.Board.from_moves([0, 3, 1, 4, 2], 3)
        green = '\x1b[1;32mX\x1b[0m'
        self.assertTrue(
            b.render(color=True).startswith(
                f'   a b c\n1 {green} {green} {green}\n'
            )
        )
        b = game.Board.from_moves([0, 27], 10, 28, 4)
        lines = b.render().splitlines()
        self.assertEqual(lines[0], '     a  b  c' + lines[0][12:])
        self.assertTrue(lines[0].endswith('  y  z aa ab'))
        self.assertTrue(lines[1].startswith(' 1  X  .'))
        self.assertTrue(lines[1].endswith('.  O'))
        self.assertTrue(lines[10].startswith('10  .'))

    def test_match_summary(self):
        res = arena.MatchResult(
            10, 3, 5, 2, {'w': 2, 'd': 2, 'l': 1}, {'w': 1, 'd': 3, 'l': 1}
        )
        self.assertEqual(
            res.summary('a', 'b'),
            'a vs b over 10 games: 3-5-2 (W-D-L), score 0.550\n'
            '  as X: 2-2-1    as O: 1-3-1',
        )

    def test_outcomes(self):
        cases = [
            ((0.0, True), 'draw'),
            ((search.WIN - 1, True), 'win in 1'),
            ((-(search.WIN - 4), True), 'loss in 2'),
            ((123.0, False), 'eval +1.23'),
        ]
        for (value, exact), want in cases:
            got = search.SearchResult(1, value, 1, 1, exact, 0.0).outcome
            self.assertEqual(got, want)

    def test_parse_move_errors(self):
        b = game.Board()
        cases = {
            '': 'empty input',
            'd1': 'd1 is off the board',
            '4,1': '4,1 is off the board',
            '9': 'index 9 is off the board',
            '?': "could not read '?' as a move",
        }
        for text, msg in cases.items():
            with self.assertRaises(game.InvalidMove) as ctx:
                b.parse_move(text)
            self.assertEqual(str(ctx.exception), msg)


def _import_gui(test):
    """Imports n_in_a_row.gui, skipping when tkinter is unavailable.

    Args:
        test: The running unittest.TestCase.

    Returns:
        The n_in_a_row.gui module.
    """
    try:
        from n_in_a_row import gui
    except (ImportError, SystemExit):
        test.skipTest('tkinter is not available')
    return gui


class TestGuiHelpers(unittest.TestCase):
    def setUp(self):
        self.gui = _import_gui(self)

    def test_solver_tags(self):
        tag = self.gui.TicTacToeGUI._solver_tag
        win = search.WIN
        cases = [
            ((0.0, True), ('D', self.gui.EVEN)),
            ((win - 1, True), ('W1', self.gui.GOOD)),
            ((-(win - 4), True), ('L2', self.gui.BAD)),
            ((1234.0, False), ('+1.2', self.gui.EVEN)),
        ]
        for (value, exact), want in cases:
            result = search.SearchResult(0, value, 1, 1, exact, 0.0)
            self.assertEqual(tag(None, result), want)

    def test_main_resolves_arguments(self):
        cases = [
            ([], (3, 3, 3, 'perfect', game.X, True, None)),
            (
                ['--size', '4', '--side', 'o', '--seed', '5'],
                (4, 4, 4, 'perfect', game.O, False, 5),
            ),
            (
                [
                    '--rows',
                    '3',
                    '--cols',
                    '5',
                    '-k',
                    '4',
                    '--solver',
                    '--opponent',
                    'greedy',
                ],
                (3, 5, 4, 'greedy', game.X, True, None),
            ),
        ]
        for argv, want in cases:
            with (
                mock.patch.object(self.gui.tk, 'Tk') as tk_cls,
                mock.patch.object(self.gui, 'TicTacToeGUI') as gui_cls,
            ):
                self.assertEqual(self.gui.main(argv), 0)
            root = tk_cls.return_value
            rows, cols, k, opponent, side, solver, seed = want
            gui_cls.assert_called_once_with(
                root,
                rows,
                cols,
                k,
                opponent_spec=opponent,
                human_side=side,
                show_solver=solver,
                seed=seed,
            )
            root.mainloop.assert_called_once_with()


class TestExamples(unittest.TestCase):
    def setUp(self):
        _require_numpy(self)

    def test_tracked_policies_pick_the_same_moves(self):
        from examples import learn_from_solver, train_policy

        positions = ([], [4], [0, 4], [0, 4, 8], [4, 0, 8])
        for module, want in (
            (train_policy, [4, 7, 7, 7, 7]),
            (learn_from_solver, [2, 0, 2, 7, 2]),
        ):
            agent = module.agent()
            got = [agent.select(game.Board.from_moves(m)) for m in positions]
            self.assertEqual(got, want, module.__name__)

    def test_short_reinforce_run(self):
        from examples import train_policy

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            policy = train_policy.train(
                40,
                'fake_agent:FirstLegal',
                3,
                3,
                3,
                hidden=8,
                lr=0.05,
                batch=8,
                seed=0,
                augment=True,
            )
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 10)
        self.assertEqual(lines[0], '       4 games   win 0.50  draw 0.00')
        self.assertEqual(lines[-1], '      40 games   win 0.17  draw 0.03')
        want = [
            0.003843,
            -0.003975,
            0.002435,
            -0.003196,
            0.00251,
            -0.00463,
            0.004477,
            -0.003711,
            0.002247,
        ]
        for got, exp in zip(policy.b2.tolist(), want):
            self.assertAlmostEqual(got, exp, places=5)
        self.assertEqual(
            train_policy.PolicyAgent(policy).select(game.Board()), 4
        )

    def test_short_distillation_run(self):
        import numpy as np

        from examples import learn_from_solver, train_policy

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            positions = learn_from_solver.collect_positions(3, 3, 3, 20, 0)
            x, m, p, _ = learn_from_solver.label(positions, None, True)
            policy = train_policy.Policy(
                x.shape[1], 8, 9, np.random.default_rng(0)
            )
            learn_from_solver.fit(
                policy, x, m, p, epochs=2, batch=64, lr=0.5, seed=0
            )
        self.assertEqual(len(positions), 119)
        self.assertEqual(x.shape, (952, 27))
        self.assertEqual(
            out.getvalue(),
            '  epoch  1  cross-entropy 1.5390\n'
            '  epoch  2  cross-entropy 1.4231\n',
        )
        want = [
            0.137636,
            -0.365757,
            0.123985,
            -0.247509,
            0.247375,
            -0.222915,
            0.287557,
            -0.108711,
            0.148339,
        ]
        for got, exp in zip(policy.b2.tolist(), want):
            self.assertAlmostEqual(got, exp, places=5)


if __name__ == '__main__':
    unittest.main()
