"""#50: the engine's explain() text must stay visible after its move.

Only that the text is shown is pinned here, not where in the label or
what else the label says, and not what happens at game end.
"""

import os
import threading
import unittest
from unittest import mock

from n_in_a_row import game

WHY = 'took the centre: n_gui_explain'


def _import_gui(test):
    try:
        from n_in_a_row import gui
    except (ImportError, SystemExit):
        if os.environ.get('CI'):
            raise
        test.skipTest('tkinter is not available')
    return gui


class _ExplainingAgent:
    def __init__(self, moves):
        self.moves = list(moves)

    def select(self, board):
        del board
        return self.moves.pop(0)

    def explain(self, board):
        del board
        return WHY


class TestEngineExplanationIsShown(unittest.TestCase):
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
        patcher = mock.patch.object(
            gui.agents, 'load_agent', return_value=_ExplainingAgent([4])
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gui = gui

    def _window_after_engine_move(self):
        # The engine (X) opens on 4; the game goes on, O to move.
        window = self.gui.TicTacToeGUI(
            self.root,
            3,
            3,
            3,
            opponent_spec='perfect',
            human_side=game.O,
            show_solver=False,
        )
        while self.workers:
            target, args, kwargs = self.workers.pop(0)
            target(*args, **kwargs)
        window._poll()
        self.assertEqual(window.board.cells[4], game.X)
        self.assertFalse(window.board.is_terminal)
        return window

    def test_explanation_is_shown_after_the_move(self):
        window = self._window_after_engine_move()
        self.assertIn(WHY, window.detail.cget('text'))

    def test_explanation_survives_a_redraw(self):
        # Moving the pointer over the board redraws it (hover), with no
        # new move: the explanation must not disappear then either.
        window = self._window_after_engine_move()
        window.draw()
        window._set_hover(0)
        self.assertIn(WHY, window.detail.cget('text'))


if __name__ == '__main__':
    unittest.main()
