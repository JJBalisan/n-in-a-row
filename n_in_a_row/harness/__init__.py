"""A shared harness for AI-safety experiments with the exact solver as truth.

Modules (all standard library except ``trainer``, which needs NumPy and is
not imported here):

    labels     exact W/D/L and mate distance per move, with an SQLite cache
    monitor    suspicion of each move from a depth-d search
    regret     per-move outcome and mate-distance regret, blunder rate
    ties       moves the solver rates exactly equal, and log2 capacity
    stats      AUROC, precision at a base rate, sample size and power, CUSUM
    zoo        saboteur, sandbagger and password-locked agents
    textplay   positions as prompts and replies as moves, for language models
    trainer    the solver-distillation pipeline as a library   [NumPy]
"""

from n_in_a_row.harness import (
    labels,
    monitor,
    regret,
    stats,
    textplay,
    ties,
    zoo,
)

__all__ = [
    "labels",
    "monitor",
    "regret",
    "stats",
    "textplay",
    "ties",
    "zoo",
]
