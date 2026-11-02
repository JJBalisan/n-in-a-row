r"""Distil the search engine into a network: supervised, fast, and it converges.

The engine can label any small position exactly, so instead of waiting for a
policy gradient to stumble onto good play we ask the solver what the right
moves are and fit them directly.  On 3x3 this reaches perfect play (never loses
to anything) in well under a minute, which makes it the better template for
your own network: swap the NumPy MLP for a torch module and keep the pipeline.
The pipeline itself lives in ``n_in_a_row.harness.trainer``; this script is
its command line.

    python3 examples/learn_from_solver.py --size 3
    python3 -m n_in_a_row match --a examples.learn_from_solver:agent \
        --b perfect --games 100
    python3 -m n_in_a_row.gui --opponent examples.learn_from_solver:agent

For boards too big to solve exactly, pass ``--depth 6``: the labels become the
best moves a depth-limited search can see, which is still a strong teacher.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import train_policy
from n_in_a_row import agents, arena, game
from n_in_a_row.harness import trainer

WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "solver_policy.npz"
)


def collect_positions(
    rows: int, cols: int, k: int, games: int, seed: int
) -> list[game.Board]:
    """Walk random games and keep every distinct position seen along the way."""
    return trainer.collect_positions(rows, cols, k, games, seed)


def label(
    positions: list[game.Board], depth: int | None, augment: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ask the engine for the optimal replies, as a probability over cells."""
    return trainer.label(positions, depth, augment, log=print)


def fit(
    policy: train_policy.Policy,
    X: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
) -> train_policy.Policy:
    """Fits the policy by masked-softmax cross-entropy on the solver's moves."""
    return trainer.fit(
        policy, X, M, P, epochs=epochs, batch=batch, lr=lr, seed=seed, log=print
    )


def agent() -> train_policy.PolicyAgent:
    """Entry point for ``--opponent examples.learn_from_solver:agent``."""
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(
            f"no trained weights at {WEIGHTS} -- run: "
            "python3 examples/learn_from_solver.py"
        )
    return train_policy.PolicyAgent(train_policy.Policy.load(WEIGHTS))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--size", type=int, default=3)
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("-k", "--k", type=int, default=None)
    p.add_argument(
        "--walks",
        type=int,
        default=3000,
        help="random games walked to gather distinct positions",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=None,
        help="teacher search depth (default: solve exactly)",
    )
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--eval-games", type=int, default=200)
    args = p.parse_args(argv)

    rows = args.rows or args.size
    cols = args.cols or rows
    k = args.k or min(rows, cols)

    print(
        f"gathering positions from {args.walks} random games on "
        f"{rows}x{cols} (k={k})"
    )
    positions = collect_positions(rows, cols, k, args.walks, args.seed)
    print(f"  {len(positions)} distinct positions\n")

    engine_name = "exact" if args.depth is None else f"depth-{args.depth}"
    print(f"labelling with the {engine_name} engine")
    X, M, P, _V = label(positions, args.depth, not args.no_augment)
    print(
        f"  {len(X)} training examples "
        f"({'no' if args.no_augment else '8x'} augmentation)\n"
    )

    rng = np.random.default_rng(args.seed)
    policy = train_policy.Policy(X.shape[1], args.hidden, rows * cols, rng)
    print("fitting")
    fit(
        policy,
        X,
        M,
        P,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
    policy.save(WEIGHTS)
    print(f"saved {WEIGHTS}\n")

    me = train_policy.PolicyAgent(policy)
    for spec in ("random", "greedy", "perfect"):
        them = agents.load_agent(spec, random.Random(0))
        games = (
            args.eval_games
            if spec != "perfect"
            else max(40, args.eval_games // 4)
        )
        print(
            arena.play_match(
                me, them, games, rows=rows, cols=cols, k=k
            ).summary("distilled", them.name)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
