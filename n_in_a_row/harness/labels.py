"""Exact labels: the game-theoretic result of every legal move in a position.

Labels come from ``Negamax(max_depth=None).evaluate_moves`` and every result
is checked for ``exact``. Positions with more than ``SOFT_EXACT_LIMIT`` empty
cells are refused rather than labelled with a horizon-limited guess.

Drawn boards are expensive (one 4x4, 4-in-a-row position with 14 empty cells
took about 200 s in the pilots), so a labeller can keep its labels in an
SQLite file keyed by the rules and ``Board.hash``. The hash is seeded from the
board shape, so it is the same in every process; each row also stores the
cells and is ignored if they differ, so a hash collision costs a search
rather than a wrong label. The file is stamped with ``CACHE_VERSION``
(SQLite's ``user_version``) and a file with another stamp is refused, so
labels from an older engine are never read back as current; bump it with
any change that could alter a label.

Each position gets a fresh engine, so a label never depends on which
positions were labelled before it. Before #12 a transposition table kept
warm across positions misreported the mate distance in 20 of the 4,520
tic-tac-toe positions; the tests check every label against brute-force
minimax.
"""

from __future__ import annotations

import json
import sqlite3
from typing import NamedTuple

from n_in_a_row import game, search

_MATE = search.WIN // 2

#: The label cache format and engine; files with another stamp are refused.
#: Unstamped files (0) were written by v1.1.0, whose labels are exact, and
#: are adopted.
CACHE_VERSION = 1


class LabelError(ValueError):
    """Raised for a position too large to label exactly."""


class InexactError(RuntimeError):
    """Raised when the engine returns a result it does not prove exact."""


class CacheVersionError(RuntimeError):
    """Raised for a label cache stamped with another ``CACHE_VERSION``."""


class MoveLabel(NamedTuple):
    """The exact result of one move, from the mover's point of view.

    Attributes:
        move: The cell index.
        value: The engine's score: ``WIN - plies`` for a win, its negative
            for a loss, 0.0 for a draw.
        outcome: 1 win, 0 draw, -1 loss.
        plies: Plies to the end of the game with best play, counting this
            move, for a win or a loss; None for a draw.
    """

    move: int
    value: float
    outcome: int
    plies: int | None


def outcome_of(value: float) -> int:
    """Maps an exact score to 1 (win), 0 (draw) or -1 (loss).

    Args:
        value: A score from the side to move's point of view.

    Returns:
        The game-theoretic outcome class.
    """
    if value > _MATE:
        return 1
    if value < -_MATE:
        return -1
    return 0


def _make_label(move: int, value: float) -> MoveLabel:
    outcome = outcome_of(value)
    plies = None if outcome == 0 else round(search.WIN - abs(value))
    return MoveLabel(move, float(value), outcome, plies)


class ExactLabeller:
    """Labels positions exactly, with an optional on-disk cache.

    Use it as a context manager, or call :meth:`close`, when a cache file is
    open. Labels are also kept in memory for the labeller's lifetime.

    Attributes:
        hits: Positions answered from the cache, memory or disk.
        searches: Positions the engine had to search.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        engine: search.Negamax | None = None,
    ) -> None:
        """Creates a labeller.

        Args:
            cache_path: An SQLite file to read and extend; None keeps
                labels in memory only.
            engine: An exact searcher to reuse for every position, for
                tests; None (the default) searches each position with a
                fresh ``Negamax(max_depth=None)``, which keeps mate
                distances exact.

        Raises:
            CacheVersionError: If the cache file has another version stamp.
        """
        self.engine = engine
        self.hits = 0
        self.searches = 0
        self._memory: dict[tuple, dict[int, MoveLabel]] = {}
        self._db: sqlite3.Connection | None = None
        if cache_path is not None:
            self._db = sqlite3.connect(cache_path)
            stamp = self._db.execute("PRAGMA user_version").fetchone()[0]
            if stamp not in (0, CACHE_VERSION):
                self.close()
                raise CacheVersionError(
                    f"{cache_path} holds version {stamp} labels, this is "
                    f"version {CACHE_VERSION}: delete it or use a new path"
                )
            # WAL commits without an fsync each, and a crash loses only the
            # newest labels; every one is worth a search, so commit each.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS labels ("
                "rows INTEGER, cols INTEGER, k INTEGER, hash INTEGER, "
                "cells TEXT, moves TEXT, "
                "PRIMARY KEY (rows, cols, k, hash))"
            )
            self._db.execute(f"PRAGMA user_version = {CACHE_VERSION}")
            self._db.commit()

    def __enter__(self) -> ExactLabeller:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Closes the cache file, if one is open."""
        if self._db is not None:
            self._db.close()
            self._db = None

    def label(self, board: game.Board) -> dict[int, MoveLabel]:
        """Returns the exact label of every legal move.

        Args:
            board: The position; it is left as it was found.

        Returns:
            One label per legal move, keyed by cell index; empty for a
            finished game.

        Raises:
            LabelError: If the position has more empty cells than
                ``search.SOFT_EXACT_LIMIT``.
            InexactError: If the engine fails to prove a result.
        """
        empties = len(board.legal_moves())
        if empties > search.SOFT_EXACT_LIMIT:
            raise LabelError(
                f"{empties} empty cells is above the exact limit of "
                f"{search.SOFT_EXACT_LIMIT}"
            )
        cells = "".join(map(str, board.cells))
        key = (*board.spec, board.hash, cells)
        found = self._memory.get(key)
        if found is None:
            found = self._read(board, cells)
        if found is not None:
            self.hits += 1
            self._memory[key] = found
            return dict(found)
        labels = self._search(board)
        self.searches += 1
        self._memory[key] = labels
        self._write(board, cells, labels)
        return dict(labels)

    def _search(self, board: game.Board) -> dict[int, MoveLabel]:
        engine = self.engine or search.Negamax(max_depth=None)
        results = engine.evaluate_moves(board)
        for move, r in results.items():
            if not r.exact:
                raise InexactError(
                    f"move {board.label(move)} came back inexact "
                    f"({r.outcome}) on {board!r}"
                )
        return {m: _make_label(m, r.value) for m, r in results.items()}

    def _read(
        self, board: game.Board, cells: str
    ) -> dict[int, MoveLabel] | None:
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT cells, moves FROM labels "
            "WHERE rows = ? AND cols = ? AND k = ? AND hash = ?",
            (*board.spec, _signed(board.hash)),
        ).fetchone()
        if row is None or row[0] != cells:
            return None
        return {m: _make_label(m, v) for m, v in json.loads(row[1])}

    def _write(
        self, board: game.Board, cells: str, labels: dict[int, MoveLabel]
    ) -> None:
        if self._db is None:
            return
        moves = json.dumps([[m, x.value] for m, x in sorted(labels.items())])
        self._db.execute(
            "INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?, ?)",
            (*board.spec, _signed(board.hash), cells, moves),
        )
        self._db.commit()


def _signed(h: int) -> int:
    """Fits a 64-bit unsigned hash into SQLite's signed INTEGER."""
    return h - (1 << 64) if h >= 1 << 63 else h


def best_outcome(labels: dict[int, MoveLabel]) -> int:
    """Returns the best outcome class any move reaches.

    Args:
        labels: The labels of a non-terminal position.

    Returns:
        1, 0 or -1.
    """
    return max(x.outcome for x in labels.values())


def optimal_moves(labels: dict[int, MoveLabel]) -> list[int]:
    """Returns the moves that keep the best outcome class, sorted.

    Mate distance is ignored here; see ``ties.tie_set`` for the moves that
    are exactly equal, distance included.

    Args:
        labels: The labels of a position.

    Returns:
        The optimal moves; empty for a finished game.
    """
    if not labels:
        return []
    best = best_outcome(labels)
    return sorted(m for m, x in labels.items() if x.outcome == best)


def blunders(labels: dict[int, MoveLabel]) -> list[int]:
    """Returns the moves that lower the outcome class, sorted.

    Args:
        labels: The labels of a position.

    Returns:
        Every move whose outcome is worse than the best available.
    """
    if not labels:
        return []
    best = best_outcome(labels)
    return sorted(m for m, x in labels.items() if x.outcome < best)


def reachable(
    rows: int, cols: int | None = None, k: int | None = None
) -> list[game.Board]:
    """Enumerates every position reachable from the empty board.

    Positions are deduplicated by their cells, so this is only practical on
    small boards (3x3 has 5,478, terminal ones included).

    Args:
        rows: Board height.
        cols: Board width; defaults to ``rows``.
        k: Stones in a row needed to win; defaults to the shorter side.

    Returns:
        One board per distinct position, finished games included.
    """
    seen: dict[tuple[int, ...], game.Board] = {}
    stack = [game.Board(rows, cols, k)]
    while stack:
        b = stack.pop()
        key = tuple(b.cells)
        if key in seen:
            continue
        seen[key] = b
        for m in b.legal_moves():
            child = b.copy()
            child.push(m)
            stack.append(child)
    return list(seen.values())
