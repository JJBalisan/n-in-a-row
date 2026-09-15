"""Distil the search engine into a network: supervised, fast, and it converges.

The engine can label any small position exactly, so instead of waiting for a
policy gradient to stumble onto good play we ask the solver what the right
moves are and fit them directly.  On 3x3 this reaches perfect play (never loses
to anything) in well under a minute, which makes it the better template for
your own network: swap the NumPy MLP for a torch module and keep the pipeline.

    python3 examples/learn_from_solver.py --size 3
    python3 -m tictactoe match --a examples.learn_from_solver:agent --b perfect --games 100
    python3 -m tictactoe.gui --opponent examples.learn_from_solver:agent

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

from examples.train_policy import Policy, PolicyAgent
from tictactoe.agents import load_agent
from tictactoe.env import dihedral, encode, legal_mask
from tictactoe.game import Board
from tictactoe.search import Negamax

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_policy.npz")


def collect_positions(rows: int, cols: int, k: int, games: int, seed: int):
    """Walk random games and keep every distinct position seen along the way."""
    rng = random.Random(seed)
    seen = {}
    for _ in range(games):
        b = Board(rows, cols, k)
        while not b.is_terminal:
            if b.hash not in seen:
                seen[b.hash] = b.copy()
            b.push(rng.choice(b.legal_moves()))
    return list(seen.values())


def label(positions, depth, augment: bool):
    """Ask the engine for the optimal replies, as a probability over cells."""
    engine = Negamax(max_depth=depth)
    xs, ms, ps, vs = [], [], [], []
    for n, b in enumerate(positions, 1):
        values = engine.evaluate_moves(b)
        if not values:
            continue
        best = max(v.value for v in values.values())
        target = np.zeros(b.size, dtype=np.float32)
        winners = [m for m, v in values.items() if v.value >= best - 1e-9]
        for m in winners:
            target[m] = 1.0 / len(winners)
        obs = encode(b)
        mask = legal_mask(b)
        value = np.sign(best) if abs(best) > 1e-9 else 0.0
        variants = (dihedral(obs, target, b.rows, b.cols) if augment
                    else [(obs, target)])
        masks = (dihedral(obs, mask.astype(np.float32), b.rows, b.cols) if augment
                 else [(obs, mask.astype(np.float32))])
        for (o_t, p_t), (_, m_t) in zip(variants, masks):
            xs.append(o_t.reshape(-1))
            ms.append(m_t.astype(bool))
            ps.append(p_t)
            vs.append(value)
        if n % 500 == 0:
            print("  labelled %d/%d positions" % (n, len(positions)))
    return (np.asarray(xs, np.float64), np.asarray(ms, bool),
            np.asarray(ps, np.float64), np.asarray(vs, np.float64))


def fit(policy: Policy, X, M, P, epochs: int, batch: int, lr: float, seed: int):
    """Plain masked-softmax cross-entropy against the solver's move distribution."""
    rng = np.random.default_rng(seed)
    n = len(X)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        loss_sum = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            x, m, target = X[idx], M[idx], P[idx]
            h, z = policy.logits(x)
            z = np.where(m, z, -1e9)
            z -= z.max(axis=-1, keepdims=True)
            e = np.exp(z) * m
            p = e / np.maximum(e.sum(-1, keepdims=True), 1e-12)

            loss_sum += -(target * np.log(np.maximum(p, 1e-12))).sum()
            dz = (target - p) / len(idx)

            dW2 = h.T @ dz
            db2 = dz.sum(0)
            dh = (dz @ policy.W2.T) * (1.0 - h * h)
            policy.W2 += lr * dW2
            policy.b2 += lr * db2
            policy.W1 += lr * (x.T @ dh)
            policy.b1 += lr * dh.sum(0)
        print("  epoch %2d  cross-entropy %.4f" % (epoch, loss_sum / n))
    return policy


def agent() -> PolicyAgent:
    """Entry point for ``--opponent examples.learn_from_solver:agent``."""
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(
            f"no trained weights at {WEIGHTS} -- run: python3 examples/learn_from_solver.py")
    return PolicyAgent(Policy.load(WEIGHTS))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--size", type=int, default=3)
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("-k", "--k", type=int, default=None)
    p.add_argument("--walks", type=int, default=3000,
                   help="random games walked to gather distinct positions")
    p.add_argument("--depth", type=int, default=None,
                   help="teacher search depth (default: solve exactly)")
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

    print("gathering positions from %d random games on %dx%d (k=%d)"
          % (args.walks, rows, cols, k))
    positions = collect_positions(rows, cols, k, args.walks, args.seed)
    print("  %d distinct positions\n" % len(positions))

    print("labelling with the %s engine" % ("exact" if args.depth is None
                                            else "depth-%d" % args.depth))
    X, M, P, _V = label(positions, args.depth, not args.no_augment)
    print("  %d training examples (%s augmentation)\n"
          % (len(X), "no" if args.no_augment else "8x"))

    rng = np.random.default_rng(args.seed)
    policy = Policy(X.shape[1], args.hidden, rows * cols, rng)
    print("fitting")
    fit(policy, X, M, P, args.epochs, args.batch, args.lr, args.seed)
    policy.save(WEIGHTS)
    print(f"saved {WEIGHTS}\n")

    from tictactoe.arena import play_match
    me = PolicyAgent(policy)
    for spec in ("random", "greedy", "perfect"):
        them = load_agent(spec, random.Random(0))
        games = args.eval_games if spec != "perfect" else max(40, args.eval_games // 4)
        print(play_match(me, them, games, rows, cols, k).summary("distilled", them.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
