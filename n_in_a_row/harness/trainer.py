"""Distil a teacher into a small network: the learn_from_solver pipeline.

Walk random games for positions, label each with a target distribution over
cells, and fit a one-hidden-layer MLP by masked-softmax cross-entropy. The
default teacher is the solver's set of best moves; pass ``target_fn`` to
:func:`dataset` to train on anything else (a locked policy, a planted
trigger, a weaker teacher).

``examples/learn_from_solver.py`` runs this pipeline and its tracked
weights load with :meth:`Policy.load`. Needs NumPy.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import numpy as np

from n_in_a_row import agents, env as rl_env, game, search

#: Receives progress lines; pass ``print`` to see them.
Log = Callable[[str], object]

#: Maps a position to (target distribution over cells, value), or None to
#: skip the position.
TargetFn = Callable[[game.Board], "tuple[np.ndarray, float] | None"]


class Policy:
    """One hidden layer. Small enough to read, big enough to solve 3x3.

    Attributes:
        W1: Input-to-hidden weights.
        b1: Hidden biases.
        W2: Hidden-to-output weights.
        b2: Output biases, one per cell.
    """

    def __init__(
        self, n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator
    ) -> None:
        """Initialises the weights (He for W1, Xavier-like for W2).

        Args:
            n_in: Input size, ``3 * rows * cols`` for an encoded board.
            n_hidden: Hidden units.
            n_out: Outputs, one per cell.
            rng: Draws the initial weights.
        """
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def logits(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runs the network.

        Args:
            x: A batch of flattened encoded boards.

        Returns:
            The hidden activations and the output logits.
        """
        h = np.tanh(x @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def probs(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Returns the move distribution with illegal cells masked out.

        Args:
            x: A batch of flattened encoded boards.
            mask: True where a move is legal, same batch shape.

        Returns:
            Probabilities over cells, zero on illegal ones.
        """
        _, z = self.logits(x)
        z = np.where(mask, z, -1e9)
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z) * mask
        return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)

    def save(self, path: str) -> None:
        """Writes the weights to an ``.npz`` file."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> Policy:
        """Reads weights written by :meth:`save`, as this class."""
        d = np.load(path)
        p = cls.__new__(cls)
        p.W1, p.b1, p.W2, p.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return p


class PolicyAgent(agents.Agent):
    """The trained net, wearing the Agent interface everything else expects.

    Attributes:
        policy: The network.
        greedy: Play the most likely move rather than sampling.
    """

    name = "policy"

    def __init__(
        self, policy: Policy, greedy: bool = True, seed: int = 0
    ) -> None:
        """Wraps a network.

        Args:
            policy: The network.
            greedy: Play the argmax rather than sampling.
            seed: Seeds the sampler.
        """
        self.policy = policy
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)

    def select(self, board: game.Board) -> int:
        """Returns the network's move."""
        x = rl_env.encode(board).reshape(1, -1).astype(np.float64)
        m = rl_env.legal_mask(board)[None, :]
        p = self.policy.probs(x, m)[0]
        if self.greedy:
            return int(np.argmax(p))
        return int(self.rng.choice(len(p), p=p))


def collect_positions(
    rows: int, cols: int, k: int, games: int, seed: int
) -> list[game.Board]:
    """Walks random games and keeps every distinct position seen.

    Args:
        rows: Board height.
        cols: Board width.
        k: Stones in a row needed to win.
        games: Random games to walk.
        seed: Seeds the walk.

    Returns:
        The distinct non-terminal positions, in the order first seen.
    """
    rng = random.Random(seed)
    seen = {}
    for _ in range(games):
        b = game.Board(rows, cols, k)
        while not b.is_terminal:
            if b.hash not in seen:
                seen[b.hash] = b.copy()
            b.push(rng.choice(b.legal_moves()))
    return list(seen.values())


def solver_targets(depth: int | None = None) -> TargetFn:
    """Makes the default teacher: uniform over the engine's best moves.

    Args:
        depth: Search depth; None solves exactly where feasible.

    Returns:
        A target function whose value is the sign of the best score.
    """
    engine = search.Negamax(max_depth=depth)

    def targets(b: game.Board) -> tuple[np.ndarray, float] | None:
        values = engine.evaluate_moves(b)
        if not values:
            return None
        best = max(v.value for v in values.values())
        target = np.zeros(b.size, dtype=np.float32)
        winners = [m for m, v in values.items() if v.value >= best - 1e-9]
        for m in winners:
            target[m] = 1.0 / len(winners)
        return target, (np.sign(best) if abs(best) > 1e-9 else 0.0)

    return targets


def dataset(
    positions: Sequence[game.Board],
    target_fn: TargetFn,
    augment: bool,
    log: Log | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds training arrays from positions and a teacher.

    Args:
        positions: The positions to label.
        target_fn: The teacher.
        augment: Add the dihedral transforms of every position (8 on a
            square board, 4 on a rectangle).
        log: Receives a progress line every 500 positions.

    Returns:
        Inputs X, legal masks M, targets P and values V, one row each per
        example.
    """
    xs, ms, ps, vs = [], [], [], []
    for n, b in enumerate(positions, 1):
        labelled = target_fn(b)
        if labelled is None:
            continue
        target, value = labelled
        obs = rl_env.encode(b)
        mask = rl_env.legal_mask(b).astype(np.float32)
        if augment:
            variants = rl_env.dihedral(obs, target, b.rows, b.cols)
            masks = rl_env.dihedral(obs, mask, b.rows, b.cols)
        else:
            variants, masks = [(obs, target)], [(obs, mask)]
        for (o_t, p_t), (_, m_t) in zip(variants, masks):
            xs.append(o_t.reshape(-1))
            ms.append(m_t.astype(bool))
            ps.append(p_t)
            vs.append(value)
        if log is not None and n % 500 == 0:
            log(f"  labelled {n}/{len(positions)} positions")
    return (
        np.asarray(xs, np.float64),
        np.asarray(ms, bool),
        np.asarray(ps, np.float64),
        np.asarray(vs, np.float64),
    )


def label(
    positions: Sequence[game.Board],
    depth: int | None,
    augment: bool,
    log: Log | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Labels positions with the solver's best moves; see :func:`dataset`.

    Args:
        positions: The positions to label.
        depth: Teacher search depth; None solves exactly.
        augment: Add the dihedral transforms of every position.
        log: Receives progress lines.

    Returns:
        X, M, P and V as :func:`dataset` returns them.
    """
    return dataset(positions, solver_targets(depth), augment, log)


def fit(
    policy: Policy,
    X: np.ndarray,  # noqa: N803 (matrix notation)
    M: np.ndarray,  # noqa: N803
    P: np.ndarray,  # noqa: N803
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    log: Log | None = None,
) -> Policy:
    """Fits the policy by masked-softmax cross-entropy on the targets.

    Args:
        policy: The network; updated in place.
        X: Inputs.
        M: Legal-move masks.
        P: Target distributions.
        epochs: Passes over the data.
        batch: Examples per gradient step.
        lr: Learning rate.
        seed: Seeds the shuffles.
        log: Receives the cross-entropy after every epoch.

    Returns:
        The same policy.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        loss_sum = 0.0
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            x, m, target = X[idx], M[idx], P[idx]
            h, z = policy.logits(x)
            z = np.where(m, z, -1e9)
            z -= z.max(axis=-1, keepdims=True)
            e = np.exp(z) * m
            p = e / np.maximum(e.sum(-1, keepdims=True), 1e-12)

            loss_sum += -(target * np.log(np.maximum(p, 1e-12))).sum()
            dz = (target - p) / len(idx)

            dw2 = h.T @ dz
            db2 = dz.sum(0)
            dh = (dz @ policy.W2.T) * (1.0 - h * h)
            policy.W2 += lr * dw2
            policy.b2 += lr * db2
            policy.W1 += lr * (x.T @ dh)
            policy.b1 += lr * dh.sum(0)
        if log is not None:
            log(f"  epoch {epoch:2d}  cross-entropy {loss_sum / n:.4f}")
    return policy


def distil(
    rows: int = 3,
    cols: int | None = None,
    k: int | None = None,
    *,
    walks: int = 3000,
    depth: int | None = None,
    target_fn: TargetFn | None = None,
    hidden: int = 128,
    epochs: int = 30,
    batch: int = 256,
    lr: float = 0.5,
    seed: int = 0,
    augment: bool = True,
    log: Log | None = None,
) -> Policy:
    """Runs the whole pipeline: walk, label, fit.

    Args:
        rows: Board height.
        cols: Board width; defaults to ``rows``.
        k: Stones in a row needed to win; defaults to the shorter side.
        walks: Random games walked to gather positions.
        depth: Teacher depth when ``target_fn`` is None.
        target_fn: The teacher; the solver's best moves if None.
        hidden: Hidden units.
        epochs: Passes over the data.
        batch: Examples per gradient step.
        lr: Learning rate.
        seed: Seeds the walk, the weights and the shuffles.
        augment: Add the dihedral transforms of every position.
        log: Receives progress lines.

    Returns:
        The trained network.
    """
    cols = cols or rows
    k = k or min(rows, cols)
    positions = collect_positions(rows, cols, k, walks, seed)
    X, M, P, _ = dataset(  # noqa: N806 (matrix notation)
        positions, target_fn or solver_targets(depth), augment, log
    )
    policy = Policy(
        X.shape[1], hidden, rows * cols, np.random.default_rng(seed)
    )
    return fit(
        policy, X, M, P, epochs=epochs, batch=batch, lr=lr, seed=seed, log=log
    )
