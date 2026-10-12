"""Negamax search with alpha-beta, a transposition table and move ordering.

Exact on small boards (3x3 solves from empty in milliseconds); depth-limited
with a threat-counting heuristic on anything larger.  A search always reports
whether its answer was exact, so callers never have to guess.
"""

from __future__ import annotations

import time
from typing import NamedTuple

from tictactoe import game

WIN = 1_000_000

#: Above this many empty cells an "exact" search silently costs minutes, so we
#: cap the depth instead and flag the result as inexact rather than hanging.
SOFT_EXACT_LIMIT = 14

EXACT, LOWER, UPPER = 0, 1, 2


class SearchResult(NamedTuple):
    move: int | None
    value: float  # from the perspective of the side to move
    depth: int
    nodes: int
    exact: bool  # True if the value is the game-theoretic result
    seconds: float

    @property
    def outcome(self) -> str:
        """'win in N', 'draw', 'loss in N' -- N counted in the mover's own moves."""
        if not self.exact:
            return "eval %+.2f" % (self.value / 100.0)
        if abs(self.value) < WIN // 2:
            return "draw"
        plies = WIN - abs(self.value)
        n = (plies + 1) // 2
        return ("win in %d" % n) if self.value > 0 else ("loss in %d" % n)


def _ordering_bias(rows: int, cols: int) -> list[float]:
    """Centre-first static ordering; cheap and surprisingly effective."""
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    out = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        out.append(-((r - cr) ** 2 + (c - cc) ** 2))
    return out


class Negamax:
    """Reusable searcher. One instance keeps its transposition table warm."""

    def __init__(
        self,
        max_depth: int | None = None,
        time_limit: float | None = None,
        tt_size: int = 1 << 20,
    ):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.tt_size = tt_size
        self._tt: dict[int, tuple[int, int, float, int | None]] = {}
        self._tt_spec: tuple[int, int, int] | None = None
        self.nodes = 0
        self._deadline = None
        self._root_ply = 0
        self._bias: list[float] = []
        self._windows = ()
        self._weights: list[float] = []

    # ---- public ---------------------------------------------------------

    def search(self, board: game.Board) -> SearchResult:
        started = time.perf_counter()
        self.nodes = 0
        self._deadline = (
            None if self.time_limit is None else started + self.time_limit
        )
        # Board.hash ignores k, so entries from other rules would be stale.
        if board.spec != self._tt_spec:
            self._tt.clear()
            self._tt_spec = board.spec
        # Mate scores count plies from *this* position, not from the empty
        # board, so "win in 1" means one move from here whatever the ply count.
        self._root_ply = board.ply
        self._bias = _ordering_bias(board.rows, board.cols)
        self._windows = game.windows(board.rows, board.cols, board.k)
        self._weights = [0.0] + [10.0**i for i in range(board.k)]

        empties = len(board.legal_moves())
        if empties == 0:
            return SearchResult(
                None, 0.0, 0, 0, True, time.perf_counter() - started
            )

        if self.max_depth is None:
            target = min(empties, SOFT_EXACT_LIMIT)
        else:
            target = min(self.max_depth, empties)

        best_move, best_value, reached = None, -float("inf"), 0
        # Iterative deepening: each pass seeds the transposition table with good
        # move ordering for the next, and a time limit can stop us mid-way with
        # a usable answer already in hand.
        for d in range(1, target + 1):
            move, value, completed = self._root(board, d)
            if move is not None:
                best_move, best_value = move, value
            if not completed:
                break
            reached = d
            if abs(value) > WIN // 2:
                break  # a forced result will not change with more depth

        # A mate-valued score means a real terminal node was reached, so it is
        # game-theoretic regardless of the horizon. A drawish score is only
        # trustworthy if we searched every remaining move.
        proven = abs(best_value) > WIN // 2 or reached >= empties

        return SearchResult(
            move=best_move,
            value=best_value,
            depth=reached,
            nodes=self.nodes,
            exact=proven,
            seconds=time.perf_counter() - started,
        )

    def evaluate_moves(self, board: game.Board) -> dict[int, SearchResult]:
        """Search every legal reply. Feeds the GUI's 'show the solver' overlay."""
        out = {}
        for m in board.legal_moves():
            board.push(m)
            try:
                if board.is_terminal:
                    value = 0.0 if board.winner == game.DRAW else float(WIN - 1)
                    out[m] = SearchResult(m, value, 0, 1, True, 0.0)
                else:
                    sub = self.search(board)
                    out[m] = sub._replace(move=m, value=shift_mate(-sub.value))
            finally:
                board.pop()
        return out

    # ---- internals ------------------------------------------------------

    def _root(self, board: game.Board, depth: int):
        alpha, beta = -float("inf"), float("inf")
        best_move, best_value = None, -float("inf")
        for m in self._ordered(board, depth):
            board.push(m)
            try:
                value = -self._negamax(board, depth - 1, -beta, -alpha)
            except _TimeUp:
                board.pop()
                return best_move, best_value, False
            board.pop()
            if value > best_value:
                best_value, best_move = value, m
            if value > alpha:  # noqa: PLR1730 (hot loop; avoids a max() call)
                alpha = value
        return best_move, best_value, True

    def _negamax(
        self, board: game.Board, depth: int, alpha: float, beta: float
    ) -> float:
        self.nodes += 1
        if self._deadline is not None and (self.nodes & 1023) == 0:
            if time.perf_counter() > self._deadline:
                raise _TimeUp()

        if board.winner is not None:
            if board.winner == game.DRAW:
                return 0.0
            # The side to move is the one who just got beaten.
            return -(WIN - (board.ply - self._root_ply))
        if depth <= 0:
            return self._eval(board)

        alpha_orig = alpha
        key = board.hash
        hit = self._tt.get(key)
        tt_move = None
        if hit is not None:
            h_depth, flag, value, tt_move = hit
            value = _mate_from_tt(value, board.ply - self._root_ply)
            if h_depth >= depth:
                if flag == EXACT:
                    return value
                if flag == LOWER and value > alpha:
                    alpha = value
                elif flag == UPPER and value < beta:
                    beta = value
                if alpha >= beta:
                    return value

        best = -float("inf")
        best_move = None
        for m in self._ordered(board, depth, tt_move):
            board.push(m)
            try:
                value = -self._negamax(board, depth - 1, -beta, -alpha)
            finally:
                board.pop()
            if value > best:
                best, best_move = value, m
            if value > alpha:  # noqa: PLR1730 (hot loop; avoids a max() call)
                alpha = value
            if alpha >= beta:
                break

        if len(self._tt) < self.tt_size:
            if best <= alpha_orig:
                flag = UPPER
            elif best >= beta:
                flag = LOWER
            else:
                flag = EXACT
            stored = _mate_to_tt(best, board.ply - self._root_ply)
            self._tt[key] = (depth, flag, stored, best_move)
        return best

    def _ordered(
        self, board: game.Board, depth: int, tt_move: int | None = None
    ) -> list[int]:
        """Immediate wins, then blocks, then stones near the action, then centre."""
        moves = board.legal_moves()
        if not moves:
            return moves
        me = board.to_move
        them = game.opponent(me)
        cells, cols, rows = board.cells, board.cols, board.rows
        empty = game.EMPTY
        bias = self._bias

        def neighbours(i: int) -> int:
            r, c = divmod(i, cols)
            n = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if (
                        0 <= rr < rows
                        and 0 <= cc < cols
                        and cells[rr * cols + cc] != empty
                    ):
                        n += 1
            return n

        def score(i: int) -> float:
            if i == tt_move:
                return 1e9
            if board.line_through_if(i, me):
                return 1e8
            if board.line_through_if(i, them):
                return 1e7
            return neighbours(i) * 10.0 + bias[i]

        moves.sort(key=score, reverse=True)
        return moves

    def _eval(self, board: game.Board) -> float:
        """Threat count: every k-window that still belongs to one player only.

        A window with c of my stones and no opponent stone is worth 10**c, so a
        line one short of complete dominates one that is two short.
        """
        cells = board.cells
        empty = game.EMPTY
        me = board.to_move
        w = self._weights
        total = 0.0
        for win in self._windows:
            mine = theirs = 0
            for i in win:
                v = cells[i]
                if v == me:
                    mine += 1
                elif v != empty:
                    theirs += 1
            if mine and theirs:
                continue
            if mine:
                total += w[mine]
            elif theirs:
                total -= w[theirs]
        return total


def shift_mate(value: float) -> float:
    """Move a mate score one ply further away (the reply we just stepped over)."""
    if value > WIN // 2:
        return value - 1
    if value < -WIN // 2:
        return value + 1
    return value


def _mate_to_tt(value: float, plies_from_root: int) -> float:
    """Re-anchors a root-relative mate score at the node being stored.

    The table outlives a single search, so a mate score must count plies from
    the stored position itself rather than from whichever root was searching.

    Args:
        value: Score relative to the current search root.
        plies_from_root: Plies between the search root and the stored node.

    Returns:
        The score with any mate distance measured from the node.
    """
    if value > WIN // 2:
        return value + plies_from_root
    if value < -WIN // 2:
        return value - plies_from_root
    return value


def _mate_from_tt(value: float, plies_from_root: int) -> float:
    """Inverts _mate_to_tt for a probe made from the current search root.

    Args:
        value: Node-relative score read from the transposition table.
        plies_from_root: Plies between the search root and the probed node.

    Returns:
        The score with any mate distance measured from the search root.
    """
    if value > WIN // 2:
        return value - plies_from_root
    if value < -WIN // 2:
        return value + plies_from_root
    return value


class _TimeUp(Exception):
    pass


def solve(board: game.Board, max_depth: int | None = None) -> SearchResult:
    """One-shot convenience wrapper."""
    return Negamax(max_depth=max_depth).search(board)
