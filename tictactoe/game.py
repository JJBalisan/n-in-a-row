"""Generalised n-in-a-row engine (tic-tac-toe on an arbitrary board).

Pure standard library on purpose: the CLI and the GUI import only this module
tree, so they run on any Python 3.9+ with nothing installed.  NumPy appears
exclusively in :mod:`tictactoe.env`, the reinforcement-learning layer.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable

EMPTY = 0
X = 1
O = 2  # noqa: E741 (public player constant)

PLAYERS = (X, O)
SYMBOL = {EMPTY: ".", X: "X", O: "O"}

DRAW = 0

#: The four axes a run can lie on: row, column, and the two diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

_ZOBRIST_CACHE = {}
_WINDOW_CACHE = {}


def opponent(player: int) -> int:
    """Returns the other player.

    Args:
        player: X or O.

    Returns:
        O for X, and X for anything else.
    """
    return O if player == X else X


class InvalidMove(ValueError):  # noqa: N818 (public name)
    """Raised for a move off the board, onto a taken cell, or after the end."""


def _zobrist(rows: int, cols: int) -> tuple[tuple[int, int, int], ...]:
    """Returns the deterministic per-shape Zobrist table.

    Seeded from the shape, so hashes match across processes.

    Args:
        rows: Board height.
        cols: Board width.

    Returns:
        One (0, x_key, o_key) triple per cell, indexed by player.
    """
    key = (rows, cols)
    table = _ZOBRIST_CACHE.get(key)
    if table is None:
        rng = random.Random((rows * 73856093) ^ (cols * 19349663) ^ 0x7CAC70E)
        table = tuple(
            (0, rng.getrandbits(64), rng.getrandbits(64))
            for _ in range(rows * cols)
        )
        _ZOBRIST_CACHE[key] = table
    return table


_SIDE_HASH = 0x9E3779B97F4A7C15


def windows(rows: int, cols: int, k: int) -> tuple[tuple[int, ...], ...]:
    """Returns every k-length straight run of cells that fits on the board.

    Cached per shape.  Used by the heuristic evaluator and by threat counting.

    Args:
        rows: Board height.
        cols: Board width.
        k: Run length.

    Returns:
        A tuple of runs, each a tuple of k cell indices.
    """
    key = (rows, cols, k)
    got = _WINDOW_CACHE.get(key)
    if got is not None:
        return got
    out = []
    for r in range(rows):
        for c in range(cols):
            for dr, dc in DIRECTIONS:
                er, ec = r + dr * (k - 1), c + dc * (k - 1)
                if not (0 <= er < rows and 0 <= ec < cols):
                    continue
                out.append(
                    tuple((r + dr * i) * cols + (c + dc * i) for i in range(k))
                )
    got = tuple(out)
    _WINDOW_CACHE[key] = got
    return got


def col_label(c: int) -> str:
    """Returns the letter label of a column: 0 -> 'a', 26 -> 'aa'.

    Args:
        c: Zero-based column index.

    Returns:
        The column's label in bijective base 26.
    """
    s = ""
    c += 1
    while c:
        c, rem = divmod(c - 1, 26)
        s = chr(ord("a") + rem) + s
    return s


def col_index(s: str) -> int:
    """Inverts :func:`col_label`.

    Args:
        s: A lower-case column label such as 'b' or 'aa'.

    Returns:
        The zero-based column index.
    """
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - ord("a") + 1)
    return n - 1


class Board:
    """A mutable position with an undo stack.

    Search pushes and pops moves rather than copying, which is what makes deep
    negamax affordable.  Anything that wants to keep a position around should
    call :meth:`copy`.

    Attributes:
        rows: Board height.
        cols: Board width.
        k: Stones in a row needed to win.
        cells: EMPTY, X or O for every cell, row by row.
        to_move: The player whose turn it is.
        winner: None while in play, DRAW, or the player who won.
        win_line: The winning run's cells once someone has won, else None.
        last_move: The most recent move, or None on an empty board.
        hash: Zobrist hash of the stones and the side to move (ignores k).
    """

    __slots__ = (
        "_history",
        "_zob",
        "cells",
        "cols",
        "hash",
        "k",
        "last_move",
        "rows",
        "to_move",
        "win_line",
        "winner",
    )

    def __init__(
        self, rows: int = 3, cols: int | None = None, k: int | None = None
    ) -> None:
        """Creates an empty board.

        Args:
            rows: Board height.
            cols: Board width; defaults to ``rows``.
            k: Stones in a row needed to win; defaults to the shorter side.

        Raises:
            ValueError: If the board is empty, k is below 1, or k is longer
                than the longest side.
        """
        cols = rows if cols is None else cols
        k = min(rows, cols) if k is None else k
        if rows < 1 or cols < 1:
            raise ValueError("board must have at least one row and column")
        if k < 1:
            raise ValueError("k must be at least 1")
        if k > max(rows, cols):
            raise ValueError(
                "k=%d cannot fit on a %dx%d board (max run length is %d)"
                % (k, rows, cols, max(rows, cols))
            )
        self.rows = rows
        self.cols = cols
        self.k = k
        self.cells: list[int] = [EMPTY] * (rows * cols)
        self.to_move = X
        self.winner: int | None = None  # None = in play, 0 = draw, X or O = won
        self.win_line: tuple[int, ...] | None = None
        self.last_move: int | None = None
        self._history: list[tuple] = []
        self._zob = _zobrist(rows, cols)
        self.hash = 0

    # ---- shape helpers -------------------------------------------------

    @property
    def size(self) -> int:
        """The number of cells."""
        return self.rows * self.cols

    @property
    def ply(self) -> int:
        """The number of stones on the board."""
        return len(self._history)

    @property
    def spec(self) -> tuple[int, int, int]:
        """The rules as a (rows, cols, k) tuple."""
        return (self.rows, self.cols, self.k)

    def coord(self, idx: int) -> tuple[int, int]:
        """Returns the zero-based (row, col) of a cell index."""
        return divmod(idx, self.cols)

    def index(self, r: int, c: int) -> int:
        """Returns the cell index of zero-based row ``r`` and column ``c``."""
        return r * self.cols + c

    def label(self, idx: int) -> str:
        """Returns a cell's name, such as 'b2'."""
        r, c = self.coord(idx)
        return "%s%d" % (col_label(c), r + 1)

    # ---- play ----------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Whether the game is over (won or drawn)."""
        return self.winner is not None

    def legal_moves(self) -> list[int]:
        """Returns the empty cells, or nothing once the game is over."""
        if self.winner is not None:
            return []
        return [i for i, v in enumerate(self.cells) if v == EMPTY]

    def legal_mask(self) -> list[bool]:
        """Returns one flag per cell: True where a move is legal."""
        if self.winner is not None:
            return [False] * self.size
        return [v == EMPTY for v in self.cells]

    def push(self, move: int) -> None:
        """Plays a stone for the side to move and updates the result.

        Args:
            move: The cell index to play.

        Raises:
            InvalidMove: If the game is over, the cell is off the board, or
                the cell is taken.
        """
        if self.winner is not None:
            raise InvalidMove("the game is already over")
        if not 0 <= move < self.size:
            raise InvalidMove(
                "move %r is off a %dx%d board" % (move, self.rows, self.cols)
            )
        if self.cells[move] != EMPTY:
            raise InvalidMove(f"{self.label(move)} is already taken")

        player = self.to_move
        self.cells[move] = player
        self.hash ^= self._zob[move][player]
        self.hash ^= _SIDE_HASH

        self._history.append((move, self.winner, self.win_line, self.last_move))
        self.last_move = move

        line = self.line_through(move, player)
        if line is not None:
            self.winner = player
            self.win_line = line
        elif len(self._history) == self.size:
            self.winner = DRAW
            self.win_line = None
        self.to_move = opponent(player)

    def pop(self) -> int:
        """Takes back the last move, restoring the previous position exactly.

        Returns:
            The cell index that was taken back.

        Raises:
            InvalidMove: If no move has been played.
        """
        if not self._history:
            raise InvalidMove("no move to undo")
        move, winner, win_line, last_move = self._history.pop()
        self.cells[move] = EMPTY
        self.to_move = opponent(self.to_move)
        self.hash ^= self._zob[move][self.to_move]
        self.hash ^= _SIDE_HASH
        self.winner = winner
        self.win_line = win_line
        self.last_move = last_move
        return move

    def line_through(self, idx: int, player: int) -> tuple[int, ...] | None:
        """Returns the winning run through ``idx`` for ``player``, if any.

        Args:
            idx: The cell the run must pass through.
            player: Whose stones count.

        Returns:
            The sorted cells of the first run of at least k found, or None.
        """
        rows, cols, k, cells = self.rows, self.cols, self.k, self.cells
        r, c = divmod(idx, cols)
        for dr, dc in DIRECTIONS:
            run = [idx]
            for sign in (1, -1):
                rr, cc = r + dr * sign, c + dc * sign
                while (
                    0 <= rr < rows
                    and 0 <= cc < cols
                    and cells[rr * cols + cc] == player
                ):
                    run.append(rr * cols + cc)
                    rr += dr * sign
                    cc += dc * sign
            if len(run) >= k:
                run.sort()
                return tuple(run)
        return None

    def winning_moves(self, player: int) -> list[int]:
        """Returns the empty cells that would complete a run for ``player``."""
        return [
            i for i in self.legal_moves() if self.line_through_if(i, player)
        ]

    def line_through_if(self, idx: int, player: int) -> tuple[int, ...] | None:
        """As :meth:`line_through`, with a ``player`` stone assumed on ``idx``.

        Args:
            idx: The cell to try.
            player: Whose stone to assume.

        Returns:
            The winning run the stone would make, or None (also None when
            the cell is already taken). The board is left unchanged.
        """
        if self.cells[idx] != EMPTY:
            return None
        self.cells[idx] = player
        try:
            return self.line_through(idx, player)
        finally:
            self.cells[idx] = EMPTY

    # ---- results -------------------------------------------------------

    def result_for(self, player: int) -> float | None:
        """Returns the result from one player's point of view.

        Args:
            player: X or O.

        Returns:
            1.0 for a win, -1.0 for a loss, 0.0 for a draw, None in play.
        """
        if self.winner is None:
            return None
        if self.winner == DRAW:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def copy(self) -> Board:
        """Returns an independent copy, undo history included."""
        b = Board.__new__(Board)
        b.rows, b.cols, b.k = self.rows, self.cols, self.k
        b.cells = list(self.cells)
        b.to_move = self.to_move
        b.winner = self.winner
        b.win_line = self.win_line
        b.last_move = self.last_move
        b._history = list(self._history)
        b._zob = self._zob
        b.hash = self.hash
        return b

    def moves_played(self) -> list[int]:
        """Returns the moves so far, in order."""
        return [h[0] for h in self._history]

    @classmethod
    def from_moves(
        cls,
        moves: Iterable[int],
        rows: int = 3,
        cols: int | None = None,
        k: int | None = None,
    ) -> Board:
        """Builds a board by playing a sequence of moves.

        Args:
            moves: Cell indices, X first.
            rows: Board height.
            cols: Board width; defaults to ``rows``.
            k: Stones in a row needed to win; defaults to the shorter side.

        Returns:
            The board after every move.

        Raises:
            InvalidMove: If any move is illegal.
        """
        b = cls(rows, cols, k)
        for m in moves:
            b.push(m)
        return b

    # ---- text I/O ------------------------------------------------------

    def parse_move(self, text: str) -> int:
        """Reads a move typed by a person.

        Args:
            text: 'b2' or 'B2', '2,3' (1-indexed row,col), or a cell index.

        Returns:
            The cell index. Whether the cell is empty is not checked.

        Raises:
            InvalidMove: If the text is empty, unreadable or off the board.
        """
        t = text.strip().lower().replace(" ", "")
        if not t:
            raise InvalidMove("empty input")
        m = re.fullmatch(r"([a-z]+)(\d+)", t)
        if m:
            c = col_index(m.group(1))
            r = int(m.group(2)) - 1
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise InvalidMove(f"{text.strip()} is off the board")
            return self.index(r, c)
        m = re.fullmatch(r"(\d+)[,;x](\d+)", t)
        if m:
            r, c = int(m.group(1)) - 1, int(m.group(2)) - 1
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise InvalidMove(f"{text.strip()} is off the board")
            return self.index(r, c)
        if t.isdigit():
            idx = int(t)
            if 0 <= idx < self.size:
                return idx
            raise InvalidMove("index %d is off the board" % idx)
        raise InvalidMove(f"could not read {text.strip()!r} as a move")

    def render(self, color: bool = False, mark_last: bool = True) -> str:
        """Draws the board as text, with column letters and row numbers.

        Args:
            color: Use ANSI colours: green for the winning run, cyan for the
                last move, dim for empty cells.
            mark_last: Highlight the last move (only with ``color``).

        Returns:
            The grid, one line per row, without a trailing newline.
        """
        width = max(
            2, len(col_label(self.cols - 1)) + 1, len(str(self.rows)) + 1
        )
        gutter = len(str(self.rows))
        highlight = set(self.win_line or ())
        lines = []
        head = " " * (gutter + 1) + "".join(
            col_label(c).rjust(width) for c in range(self.cols)
        )
        lines.append(head)
        for r in range(self.rows):
            row = [str(r + 1).rjust(gutter)]
            for c in range(self.cols):
                i = self.index(r, c)
                ch = SYMBOL[self.cells[i]]
                if color:
                    if i in highlight:
                        ch = f"\033[1;32m{ch}\033[0m"
                    elif mark_last and i == self.last_move:
                        ch = f"\033[1;36m{ch}\033[0m"
                    elif self.cells[i] == EMPTY:
                        ch = f"\033[2m{ch}\033[0m"
                row.append(ch.rjust(width + (len(ch) - 1 if color else 0)))
            lines.append("".join(row))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render(color=False)

    def __repr__(self) -> str:
        return "<Board %dx%d k=%d ply=%d to_move=%s>" % (
            self.rows,
            self.cols,
            self.k,
            self.ply,
            SYMBOL[self.to_move],
        )
