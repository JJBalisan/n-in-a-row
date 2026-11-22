"""#63: a huge number typed in a size spinbox must not be parsed whole.

Tcl's integer parse is quadratic in the digit count (200,000 digits take
seconds on the Tk thread). The check counts what reaches the Tcl
interpreter, never seconds. Whether New game then shows the dialog or
clamps to 15 is left open: only the parse of the long text is pinned.
"""

import os
import threading
import unittest
from unittest import mock

from n_in_a_row import game

LONG = '7' * 5000


def _import_gui(test):
    try:
        from n_in_a_row import gui
    except (ImportError, SystemExit):
        if os.environ.get('CI'):
            raise
        test.skipTest('tkinter is not available')
    return gui


class _Recording:
    """Forwards to a Tcl interpreter, recording every string argument."""

    def __init__(self, real, seen):
        self._real = real
        self._seen = seen

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            for arg in list(args) + list(kwargs.values()):
                self._seen.append((name, str(arg)))
            return attr(*args, **kwargs)

        return call


class _Root:
    """A Tk root whose interpreter is the recording one."""

    def __init__(self, real, tk):
        self._real = real
        self.tk = tk

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestLongNumberIsNotParsed(unittest.TestCase):
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

        fake = mock.Mock()
        fake.Lock = threading.Lock
        fake.Thread.return_value = mock.Mock()
        patcher = mock.patch.object(gui, 'threading', fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(gui.messagebox, 'showerror')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gui = gui

    def _window(self):
        window = self.gui.TicTacToeGUI(
            self.root,
            3,
            3,
            3,
            opponent_spec='human',
            human_side=game.X,
            show_solver=False,
        )
        seen = []
        recording = _Recording(self.root.tk, seen)
        # Every path from a control's text to a number goes through one of
        # these: the window's root, its interpreter, or a variable's.
        window.root = _Root(self.root, recording)
        for var in (window.rows_var, window.cols_var, window.k_var):
            var._tk = recording
        return window, seen

    def test_long_number_never_reaches_a_numeric_parse(self):
        for name in ('rows_var', 'cols_var', 'k_var'):
            with self.subTest(control=name):
                window, seen = self._window()
                self.root.globalsetvar(str(getattr(window, name)), LONG)
                window.new_game()
                parsed = [
                    method
                    for method, arg in seen
                    if LONG in arg and method != 'globalgetvar'
                ]
                self.assertEqual(parsed, [])

    def test_ordinary_size_still_plays(self):
        # Control: a short number still sets the board.
        window, _ = self._window()
        window.rows_var.set(5)
        window.new_game()
        self.assertEqual(window.board.spec, (5, 3, 3))


if __name__ == '__main__':
    unittest.main()
