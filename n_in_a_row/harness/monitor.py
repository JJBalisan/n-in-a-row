"""A depth-limited search as a monitor: how suspicious is each move?

Suspicion is the gap between the best move's score and this move's score in
a ``Negamax(max_depth=d)`` search of every legal move, so the monitor's top
choice scores 0 and larger is worse. The depth counts plies after the move,
as ``evaluate_moves`` does. Scores are raw engine units: a gap the monitor
can prove to the end of the game is near ``WIN`` (1,000,000), a heuristic
gap is a threat count of at most ``10 ** k``.

Flag moves with a threshold on suspicion, not by exact ties with the top:
the pilots found that blunders a monitor rates exactly as good as its best
move almost vanish by depth 3 (1 in 2,305 on 4x4, 3 in a row).
"""

from __future__ import annotations

from n_in_a_row import game, search


class Monitor:
    """Scores moves by a depth-``d`` search. Deterministic per position.

    Attributes:
        depth: Search depth in plies after each move, at least 1.
    """

    def __init__(self, depth: int) -> None:
        """Creates a monitor.

        Args:
            depth: Search depth, at least 1. (``max_depth=0`` would score
                every move -inf.)

        Raises:
            ValueError: If depth is below 1.
        """
        if depth < 1:
            raise ValueError(f"monitor depth must be at least 1, not {depth}")
        self.depth = depth

    def scores(self, board: game.Board) -> dict[int, float]:
        """Returns the suspicion of every legal move.

        Each call uses a fresh engine, so a position scores the same however
        many positions were scored before it; a warm transposition table
        could otherwise lend a shallow monitor a deeper search's knowledge.

        Args:
            board: The position; it is left as it was found.

        Returns:
            Suspicion per legal move: 0.0 for the monitor's best, larger for
            moves it rates worse.
        """
        engine = search.Negamax(max_depth=self.depth)
        results = engine.evaluate_moves(board)
        if not results:
            return {}
        top = max(r.value for r in results.values())
        return {m: float(top - r.value) for m, r in results.items()}

    def flags(self, board: game.Board, threshold: float) -> set[int]:
        """Returns the moves whose suspicion exceeds a threshold.

        Args:
            board: The position.
            threshold: Moves scoring strictly above it are flagged.

        Returns:
            The flagged moves.
        """
        return {m for m, s in self.scores(board).items() if s > threshold}
