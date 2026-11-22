"""n-gui s3: boundary cases for the #34, #50 and #63 fixes."""

import os
import threading
import unittest
from unittest import mock

from n_in_a_row import game


def _import_gui(test):
    try:
        from n_in_a_row import gui
    except (ImportError, SystemExit):
        if os.environ.get('CI'):
            raise
        test.skipTest('tkinter is not available')
    return gui


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
        patcher = mock.patch.object(gui.messagebox, 'showerror')
        self.showerror = patcher.start()
        self.addCleanup(patcher.stop)
        self.gui = gui

    def _window(self, spec='human', side=game.X, solver=False, size=3):
        return self.gui.TicTacToeGUI(
            self.root,
            size,
            size,
            3,
            opponent_spec=spec,
            human_side=side,
            show_solver=solver,
        )

    def _drain(self, window):
        while self.workers:
            target, args, kwargs = self.workers.pop(0)
            target(*args, **kwargs)
        window._poll()


class _Agent:
    def select(self, board):
        return board.legal_moves()[0]

    def explain(self, board):
        del board
        return 'WHY-s3'


class TestNumberBoundary(_Base):
    def _try(self, text):
        window = self._window()
        self.root.globalsetvar(str(window.rows_var), text)
        return window

    def test_long_text_is_refused_with_the_dialog(self):
        window = self._try('9' * 13)
        window.new_game()
        self.showerror.assert_called_once()
        self.assertEqual(window.board.spec, (3, 3, 3))

    def test_padded_short_number_still_parses(self):
        window = self._try('  7  ')
        window.new_game()
        self.assertEqual(window.board.spec, (7, 3, 3))

    def test_twelve_chars_is_still_parsed_and_clamped(self):
        window = self._try('0' * 10 + '40')
        window.new_game()
        self.assertEqual(window.board.spec[0], 15)

    def test_non_numbers_still_show_the_dialog(self):
        for text in ('', 'abc', '4.9'):
            with self.subTest(text=text):
                window = self._try(text)
                window.new_game()
                self.showerror.assert_called_once()
                self.showerror.reset_mock()


class TestExplanationBoundary(_Base):
    def _after_engine_move(self):
        with mock.patch.object(
            self.gui.agents, 'load_agent', return_value=_Agent()
        ):
            window = self._window('perfect', game.O)
        self._drain(window)
        self.assertEqual(window.board.ply, 1)
        return window

    def test_explanation_goes_when_the_position_changes(self):
        window = self._after_engine_move()
        self.assertIn('WHY-s3', window.detail.cget('text'))
        window.board.push(window.board.legal_moves()[0])
        window.draw()
        self.assertNotIn('WHY-s3', window.detail.cget('text'))

    def test_explanation_goes_on_new_game(self):
        window = self._after_engine_move()
        with mock.patch.object(
            self.gui.agents, 'load_agent', return_value=_Agent()
        ):
            window.new_game()
        self.workers.clear()
        window.draw()
        self.assertNotIn('WHY-s3', window.detail.cget('text'))

    def test_empty_explanation_shows_the_size_line(self):
        window = self._window()
        window.why = ''
        window.draw()
        self.assertIn('3x3', window.detail.cget('text'))


class TestOverlayBoundary(_Base):
    def test_read_for_replaced_game_with_same_board_is_dropped(self):
        # Same spec and hash (empty board), but a new game since.
        window = self._window(solver=True)
        (stale,) = self.workers
        self.workers.clear()
        window.new_game(keep_position=True)
        self.workers.clear()
        target, args, kwargs = stale
        target(*args, **kwargs)
        window._poll()
        self.assertEqual(window.solver_values, {})

    def test_read_uses_the_board_it_was_asked_for(self):
        window = self._window(solver=True)
        (live,) = self.workers
        self.workers.clear()
        target, args, kwargs = live
        window.board.push(0)  # the board moves on before the worker runs
        target(*args, **kwargs)
        window._poll()
        self.assertEqual(window.solver_values, {})


if __name__ == '__main__':
    unittest.main()
