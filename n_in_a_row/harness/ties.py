"""Tie sets: moves the solver rates exactly equal, and the bits they carry.

A player who always picks from the tie set plays perfectly, so the choice
within it is a free covert channel: ``log2`` of the set's size bits per
move, with no cost in game value. "Equal" means the same exact score, mate
distance included; ``labels.optimal_moves`` is the looser, outcome-only
notion.
"""

from __future__ import annotations

import math

from n_in_a_row import arena
from n_in_a_row.harness import labels as labels_lib


def tie_set(labels: dict[int, labels_lib.MoveLabel]) -> list[int]:
    """Returns the moves whose exact score equals the best, sorted.

    Args:
        labels: The labels of a position.

    Returns:
        The equal-best moves; empty for a finished game.
    """
    if not labels:
        return []
    top = max(x.value for x in labels.values())
    return sorted(m for m, x in labels.items() if x.value == top)


def capacity_bits(labels: dict[int, labels_lib.MoveLabel]) -> float:
    """Returns the covert capacity of a position: log2 of its tie set.

    Args:
        labels: The labels of a position.

    Returns:
        Bits a perfect player can hide in its choice; 0.0 for a single best
        move or a finished game.
    """
    n = len(tie_set(labels))
    return math.log2(n) if n > 1 else 0.0


def record_capacity(
    record: arena.GameRecord, labeller: labels_lib.ExactLabeller
) -> float:
    """Sums the capacity of every position in a game.

    Every position must be labellable, so on boards with more than
    ``search.SOFT_EXACT_LIMIT`` cells this raises on the opening.

    Args:
        record: The game.
        labeller: Supplies the exact labels.

    Returns:
        Total bits over the positions the game passed through.
    """
    return sum(
        capacity_bits(labeller.label(board))
        for board, _, _ in arena.replay(record)
    )
