"""#34: an overlay read must only land on the board it was computed for.

Every empty board hashes to 0, and k is not in the hash, so a read
matched by ``Board.hash`` alone lands on a board of another shape or k.
"""

import os
import threading
import unittest
from unittest import mock

from n_in_a_row import game, search


def _import_gui(test):
    try:
        from n_in_a_row import gui
    except (ImportError, SystemExit):
        if os.environ.get('CI'):
            raise
        test.skipTest('tkinter is not available')
    return gui


def _marker(spec):
    """A value that names the board spec it was computed for."""
    rows, cols, k = spec
    return float(rows * 10000 + cols * 100 + k)


class TestOverlayReadsLandOnTheirBoard(unittest.TestCase):
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
        # Hold each worker instead of starting it, to run it when we choose.
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
        self.gui = gui
        # Each read answers instantly with values naming its board's spec;
        # on_read, when set, runs inside the first read (mid-search).
        self.on_read = None
        test = self

        def evaluate_moves(engine, board):
            del engine
            hook, test.on_read = test.on_read, None
            if hook is not None:
                hook()
            value = _marker(board.spec)
            return {
                i: search.SearchResult(i, value, 1, 1, True, 0.0)
                for i in board.legal_moves()
            }

        patcher = mock.patch.object(
            gui.search.Negamax, 'evaluate_moves', evaluate_moves
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _window(self, rows, cols, k):
        return self.gui.TicTacToeGUI(
            self.root,
            rows,
            cols,
            k,
            opponent_spec='human',
            human_side=game.X,
            show_solver=True,
        )

    def _run(self, worker):
        target, args, kwargs = worker
        target(*args, **kwargs)

    def _new_game(self, window, rows, cols, k):
        window.rows_var.set(rows)
        window.cols_var.set(cols)
        window.k_var.set(k)
        window.new_game()

    def _values(self, window):
        return {i: r.value for i, r in window.solver_values.items()}

    def test_stale_read_after_k_change_does_not_replace_live_one(self):
        # The issue's scenario: a 3x3 k=3 read finishes after New game with
        # k=2, after the k=2 read has already landed.
        window = self._window(3, 3, 3)
        (stale,) = self.workers
        self.workers.clear()

        def mid_search():
            self._new_game(window, 3, 3, 2)
            (live,) = self.workers
            self._run(live)

        self.on_read = mid_search
        self._run(stale)
        window._poll()
        self.assertEqual(window.board.spec, (3, 3, 2))
        want = {i: _marker((3, 3, 2)) for i in range(9)}
        self.assertEqual(self._values(window), want)

    def test_stale_read_is_not_applied_to_a_new_shape(self):
        # A 3x3 read must not tag cells of an empty 4x4 board (same k).
        window = self._window(3, 3, 3)
        (stale,) = self.workers
        self.workers.clear()
        self.on_read = lambda: self._new_game(window, 4, 4, 3)
        self._run(stale)
        window._poll()
        self.assertEqual(window.board.spec, (4, 4, 3))
        self.assertEqual(self._values(window), {})

    def test_stale_read_is_not_applied_after_k_change(self):
        window = self._window(3, 3, 3)
        (stale,) = self.workers
        self.workers.clear()
        self.on_read = lambda: self._new_game(window, 3, 3, 2)
        self._run(stale)
        window._poll()
        self.assertEqual(window.board.spec, (3, 3, 2))
        self.assertEqual(self._values(window), {})

    def test_live_read_still_lands(self):
        # Control: a read for the board on screen is shown.
        window = self._window(4, 4, 3)
        (live,) = self.workers
        self._run(live)
        window._poll()
        want = {i: _marker((4, 4, 3)) for i in range(16)}
        self.assertEqual(self._values(window), want)


if __name__ == '__main__':
    unittest.main()
