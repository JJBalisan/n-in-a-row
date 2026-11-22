"""Selector's adversarial checks for #34, #50 and #63 (n-gui lane).

Written from the issue text and the verifier's notes, before reading any
solver's diff. They pin behaviour around the fixes that the verifier's
checks leave open: a stale read within one game, what the engine's
explanation does once its move is gone, and short or odd spinbox text.
"""

import os
import threading
import unittest
from unittest import mock

from n_in_a_row import game, search

WHY = 'blocked the row: n_gui_select'
LONG = '7' * 5000


def _import_gui(test):
    try:
        from n_in_a_row import gui
    except (ImportError, SystemExit):
        if os.environ.get('CI'):
            raise
        test.skipTest('tkinter is not available')
    return gui


def _marker(board):
    """A value naming the exact position a read was computed for."""
    rows, cols, k = board.spec
    return float(rows * 10000 + cols * 100 + k) + board.ply / 100.0


class _Base(unittest.TestCase):
    def setUp(self):
        gui = _import_gui(self)
        try:
            self.root = gui.tk.Tk()
        except gui.tk.TclError:
            if os.environ.get('CI'):
                raise
            self.skipTest('no display')
        self.addCleanup(self.root.destroy)
        self.root.withdraw()
        self.workers = []

        def thread(target, args=(), kwargs=None, daemon=None, **_):
            self.workers.append((target, tuple(args), dict(kwargs or {})))
            return mock.Mock()

        fake = mock.Mock()
        fake.Lock = threading.Lock
        fake.Thread.side_effect = thread
        patcher = mock.patch.object(gui, 'threading', fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.showerror = mock.Mock()
        patcher = mock.patch.object(gui.messagebox, 'showerror', self.showerror)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gui = gui

    def _run(self, worker):
        target, args, kwargs = worker
        target(*args, **kwargs)

    def _run_all(self):
        while self.workers:
            self._run(self.workers.pop(0))


class TestOverlayWithinOneGame(_Base):
    """#34: the tag must still tell positions of one game apart."""

    def setUp(self):
        super().setUp()

        # on_read, when set, runs inside the next read (mid-search).
        self.on_read = None
        test = self

        def evaluate_moves(engine, board):
            del engine
            hook, test.on_read = test.on_read, None
            if hook is not None:
                hook()
            value = _marker(board)
            return {
                i: search.SearchResult(i, value, 1, 1, True, 0.0)
                for i in board.legal_moves()
            }

        patcher = mock.patch.object(
            self.gui.search.Negamax, 'evaluate_moves', evaluate_moves
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _window(self, rows=3, cols=3, k=3):
        return self.gui.TicTacToeGUI(
            self.root,
            rows,
            cols,
            k,
            opponent_spec='human',
            human_side=game.X,
            show_solver=True,
        )

    def _play(self, window, idx):
        # What on_click does, without a pointer event.
        window.board.push(idx)
        window.solver_values = {}
        window.draw()
        window.after_move()

    def _values(self, window):
        return {i: r.value for i, r in window.solver_values.items()}

    def test_read_from_before_a_click_is_dropped(self):
        # Same game, same spec: only the position differs. A tag of
        # generation and spec alone would show the empty board's values.
        window = self._window()
        (stale,) = self.workers
        self.workers.clear()
        self.on_read = lambda: self._play(window, 0)
        self._run(stale)
        window._poll()
        self.assertEqual(window.board.ply, 1)
        self.assertEqual(self._values(window), {})

    def test_live_read_after_a_click_lands(self):
        window = self._window()
        (stale,) = self.workers
        self.workers.clear()
        self.on_read = lambda: self._play(window, 0)
        self._run(stale)
        (live,) = self.workers
        self._run(live)
        window._poll()
        want = {i: _marker(window.board) for i in range(1, 9)}
        self.assertEqual(self._values(window), want)

    def test_stale_and_live_in_one_poll(self):
        # Both reads queued before one poll: only the live one shows,
        # whatever order they arrive in.
        for live_first in (True, False):
            with self.subTest(live_first=live_first):
                self.workers.clear()
                window = self._window(4, 4, 4)
                (stale,) = self.workers
                self.workers.clear()

                def mid_search(window=window, live_first=live_first):
                    window.rows_var.set(4)
                    window.cols_var.set(4)
                    window.k_var.set(3)
                    window.new_game()
                    if live_first:
                        self._run(self.workers.pop())

                self.on_read = mid_search
                self._run(stale)
                if not live_first:
                    self._run(self.workers.pop())
                window._poll()
                self.assertEqual(window.board.spec, (4, 4, 3))
                want = {i: _marker(window.board) for i in range(16)}
                self.assertEqual(self._values(window), want)

    def test_overlay_restart_read_lands(self):
        # Toggling the overlay restarts the read (keep_position): the
        # read started by the restart must still be shown.
        window = self._window()
        self.workers.clear()
        window.show_solver.set(False)
        window.show_solver.set(True)
        self.assertTrue(self.workers)
        live = self.workers[-1]
        self.workers.clear()
        self._run(live)
        window._poll()
        want = {i: _marker(window.board) for i in range(9)}
        self.assertEqual(self._values(window), want)


class _ExplainingAgent:
    def __init__(self, moves, why=WHY):
        self.moves = list(moves)
        self.why = why

    def select(self, board):
        del board
        return self.moves.pop(0)

    def explain(self, board):
        del board
        return self.why


class TestExplanationLifetime(_Base):
    """#50: the explanation shows, and does not outlive its move."""

    def _window(self, agent):
        patcher = mock.patch.object(
            self.gui.agents, 'load_agent', return_value=agent
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        window = self.gui.TicTacToeGUI(
            self.root,
            3,
            3,
            3,
            opponent_spec='perfect',
            human_side=game.O,
            show_solver=False,
        )
        self._run_all()
        window._poll()
        return window

    def test_explanation_is_shown(self):
        window = self._window(_ExplainingAgent([4]))
        self.assertIn(WHY, window.detail.cget('text'))

    def test_new_game_clears_the_explanation(self):
        window = self._window(_ExplainingAgent([4]))
        self.assertIn(WHY, window.detail.cget('text'))
        window.side_var.set('X')  # the human opens, the engine waits
        window.new_game()
        self._run_all()
        window._poll()
        self.assertEqual(window.board.ply, 0)
        self.assertNotIn(WHY, window.detail.cget('text'))

    def test_human_move_and_undo_clear_the_explanation(self):
        # Differential (solvers clear it, or hide it by ply): once the
        # move it explains is answered or taken back, it must go.
        window = self._window(_ExplainingAgent([4, 0], why=WHY))
        self.assertIn(WHY, window.detail.cget('text'))
        window.board.push(8)  # what on_click does for O on 8
        window.solver_values = {}
        window.draw()
        window.after_move()
        self._run_all()
        window._poll()
        self.assertEqual(window.board.ply, 3)
        self.assertIn(WHY, window.detail.cget('text'))
        window.undo()
        self.assertEqual(window.board.ply, 1)
        self._run_all()
        window._poll()
        self.assertNotIn(WHY, window.detail.cget('text'))

    def test_click_clears_the_explanation(self):
        window = self._window(_ExplainingAgent([4]))
        self.assertIn(WHY, window.detail.cget('text'))
        event = mock.Mock(x=0, y=0)
        with (
            mock.patch.object(window, 'cell_at', return_value=0),
            mock.patch.object(window, 'after_move'),
        ):
            window.on_click(event)
        self.assertEqual(window.board.ply, 2)
        self.assertNotIn(WHY, window.detail.cget('text'))

    def test_no_explanation_shows_nothing_odd(self):
        # An engine with nothing to say leaves the usual detail text.
        for why in (None, ''):
            with self.subTest(why=why):
                window = self._window(_ExplainingAgent([4], why=why))
                text = window.detail.cget('text')
                self.assertNotIn('None', text)
                self.assertIn('3x3', text)

    def test_human_game_detail_unchanged(self):
        window = self.gui.TicTacToeGUI(
            self.root,
            3,
            3,
            3,
            opponent_spec='human',
            human_side=game.X,
            show_solver=False,
        )
        self.assertEqual(window.detail.cget('text'), '3x3, 3 in a row')


class TestSpinboxText(_Base):
    """#63: the guard must not refuse ordinary or merely large sizes."""

    def _window(self):
        return self.gui.TicTacToeGUI(
            self.root,
            3,
            3,
            3,
            opponent_spec='human',
            human_side=game.X,
            show_solver=False,
        )

    def _new_game_with(self, window, name, text):
        self.root.globalsetvar(str(getattr(window, name)), text)
        window.new_game()

    def test_big_but_short_numbers_still_clamp(self):
        # Base parsed and clamped these; Tcl reads a dozen characters at
        # once, so the guard has no reason to refuse them. The dialog
        # would also be wrong: it says they are not whole numbers.
        for text in ('100', '1000', '99999', '1000000', '0000000040'):
            with self.subTest(text=text):
                self.showerror.reset_mock()
                window = self._window()
                self._new_game_with(window, 'rows_var', text)
                self.showerror.assert_not_called()
                self.assertEqual(window.board.spec, (15, 3, 3))

    def test_padded_number_still_accepted(self):
        # Tcl's parse took padded and zero-led numbers before the fix;
        # keep taking them.
        for text in (' 5 ', '   5   ', '0000005', '\t5\n'):
            with self.subTest(text=repr(text)):
                self.showerror.reset_mock()
                window = self._window()
                self._new_game_with(window, 'cols_var', text)
                self.showerror.assert_not_called()
                self.assertEqual(window.board.spec, (3, 5, 3))

    def test_long_text_leaves_a_consistent_game(self):
        # Dialog with the game untouched, or a clamp to 15: either is
        # fine, a half-changed game is not.
        cases = (
            (LONG, 15),
            ('-' + LONG, 3),
            (' ' * 5000 + '5', 5),
            ('0' * 5000 + '5', 5),
        )
        for text, clamped in cases:
            with self.subTest(text=text[:3] + '...'):
                self.showerror.reset_mock()
                window = self._window()
                before = window.generation
                self._new_game_with(window, 'rows_var', text)
                if self.showerror.called:
                    self.assertEqual(window.generation, before)
                    self.assertEqual(window.board.spec, (3, 3, 3))
                else:
                    self.assertEqual(window.board.spec, (clamped, 3, 3))


if __name__ == '__main__':
    unittest.main()
