"""Exact move values by plain minimax, the reference the engine tests use.

No pruning and no transposition table, so it shares no code with
``search.Negamax``; positions are memoised by their cells. Practical only
on boards as small as tic-tac-toe.
"""

from n_in_a_row import game, search

_MINIMAX = {}


def move_values(board):
    """Returns {move: exact score} for every legal move, mate distance in."""
    out = {}
    for m in board.legal_moves():
        board.push(m)
        out[m] = search.shift_mate(-_minimax(board))
        board.pop()
    return out


def _minimax(board):
    if board.is_terminal:
        # The side to move has just lost, or nobody has won.
        return 0.0 if board.winner == game.DRAW else -float(search.WIN)
    key = tuple(board.cells)
    if key not in _MINIMAX:
        _MINIMAX[key] = max(move_values(board).values())
    return _MINIMAX[key]
