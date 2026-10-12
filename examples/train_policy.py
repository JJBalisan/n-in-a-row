"""A worked example: train a policy network against the engine, then plug it back in.

Deliberately NumPy-only so it runs with no install. It is a plain REINFORCE
policy gradient -- not a strong method, just the shortest honest path from
"empty board" to "an agent the CLI and the GUI will play against".

    python3 examples/train_policy.py --games 40000 --opponent greedy
    python3 -m tictactoe match --a examples.train_policy:agent --b greedy --games 400
    python3 -m tictactoe.gui --opponent examples.train_policy:agent

The two things worth copying into a real PyTorch setup:

* mask the illegal moves before the softmax (``obs`` and ``mask`` come straight
  from :mod:`tictactoe.env`), and
* train on the canonical observation, so one network plays both colours.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tictactoe import agents, arena, env as rl_env

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.npz")


class Policy:
    """One hidden layer. Small enough to read, big enough to solve 3x3."""

    def __init__(
        self, n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator
    ):
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def logits(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def probs(self, x, mask):
        _, z = self.logits(x)
        z = np.where(mask, z, -1e9)
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z) * mask
        return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)

    def step(
        self,
        x,
        mask,
        actions,
        advantage,
        lr: float,
        entropy: float = 0.01,
        l2: float = 1e-5,
    ):
        h, z = self.logits(x)
        z = np.where(mask, z, -1e9)
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z) * mask
        p = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)

        dz = -p
        dz[np.arange(len(actions)), actions] += 1.0
        dz *= advantage[:, None]  # REINFORCE: grad log pi(a|s) * A

        if entropy:
            # Without this the policy commits to one opening far too early and
            # stops seeing the rest of the board.
            logp = np.log(np.maximum(p, 1e-12))
            H = -(p * logp).sum(axis=-1, keepdims=True)
            dz += entropy * (-p * (logp + H)) * mask

        dz /= len(actions)

        dW2 = h.T @ dz - l2 * self.W2
        db2 = dz.sum(0)
        dh = (dz @ self.W2.T) * (1.0 - h * h)
        dW1 = x.T @ dh - l2 * self.W1
        db1 = dh.sum(0)

        self.W2 += lr * dW2
        self.b2 += lr * db2
        self.W1 += lr * dW1
        self.b1 += lr * db1

    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> Policy:
        d = np.load(path)
        p = cls.__new__(cls)
        p.W1, p.b1, p.W2, p.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return p


def rollout(
    policy: Policy,
    env: rl_env.TicTacToeEnv,
    rng: np.random.Generator,
    augment: bool,
):
    """One game. Returns (observations, masks, actions, final reward)."""
    obs = env.reset(swap_sides=bool(rng.integers(2)))
    xs, ms, acts = [], [], []
    done, reward = False, 0.0
    while not done:
        mask = env.legal_mask()
        flat = obs.reshape(-1)
        p = policy.probs(flat[None, :], mask[None, :])[0]
        a = int(rng.choice(len(p), p=p))
        xs.append(flat)
        ms.append(mask)
        acts.append(a)
        obs, reward, done, _ = env.step(a)
    if augment:
        xs, ms, acts = _augment(xs, ms, acts, env)
    return xs, ms, acts, reward


def _augment(xs, ms, acts, env):
    """Reflect and rotate every state. Eight boards for the price of one."""
    shape = env.observation_shape
    out_x, out_m, out_a = [], [], []
    for x, m, a in zip(xs, ms, acts):
        onehot = np.zeros(env.n_actions, dtype=np.float32)
        onehot[a] = 1.0
        obs = x.reshape(shape)
        variants = rl_env.dihedral(obs, onehot, env.rows, env.cols)
        masks = rl_env.dihedral(obs, m.astype(np.float32), env.rows, env.cols)
        for (obs_t, pol_t), (_, mask_t) in zip(variants, masks):
            out_x.append(obs_t.reshape(-1))
            out_m.append(mask_t.astype(bool))
            out_a.append(int(np.argmax(pol_t)))
    return out_x, out_m, out_a


def train(
    games: int,
    opponent_spec: str,
    rows: int,
    cols: int,
    k: int,
    hidden: int,
    lr: float,
    batch: int,
    seed: int,
    augment: bool,
    entropy: float = 0.01,
) -> Policy:
    rng = np.random.default_rng(seed)
    env = rl_env.TicTacToeEnv(
        rows, cols, k, opponent=agents.load_agent(opponent_spec), seed=seed
    )
    n_in = int(np.prod(env.observation_shape))
    policy = Policy(n_in, hidden, env.n_actions, rng)

    xs, ms, acts, rets = [], [], [], []
    played = won = drawn = 0
    for g in range(1, games + 1):
        gx, gm, ga, reward = rollout(policy, env, rng, augment)
        xs.extend(gx)
        ms.extend(gm)
        acts.extend(ga)
        rets.extend([reward] * len(ga))
        played += 1
        won += reward > 0
        drawn += reward == 0

        if played >= batch:
            X_ = np.asarray(xs, dtype=np.float64)
            M_ = np.asarray(ms, dtype=bool)
            A_ = np.asarray(acts, dtype=np.int64)
            R_ = np.asarray(rets, dtype=np.float64)
            adv = R_ - R_.mean()
            sd = adv.std()
            if sd > 1e-8:
                adv /= sd
            policy.step(X_, M_, A_, adv, lr, entropy)
            xs, ms, acts, rets = [], [], [], []
            played = 0

        if g % max(1, games // 10) == 0:
            print("  %6d games   win %.2f  draw %.2f" % (g, won / g, drawn / g))
    return policy


def evaluate(
    policy: Policy, spec: str, games: int, rows: int, cols: int, k: int
) -> str:

    me = PolicyAgent(policy)
    them = agents.load_agent(spec)
    res = arena.play_match(me, them, games, rows, cols, k)
    return res.summary("policy", them.name)


class PolicyAgent(agents.Agent):
    """The trained net, wearing the Agent interface the rest of the code expects."""

    name = "policy"

    def __init__(self, policy: Policy, greedy: bool = True, seed: int = 0):
        self.policy = policy
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)

    def select(self, board) -> int:

        x = rl_env.encode(board).reshape(1, -1).astype(np.float64)
        m = rl_env.legal_mask(board)[None, :]
        p = self.policy.probs(x, m)[0]
        if self.greedy:
            return int(np.argmax(p))
        return int(self.rng.choice(len(p), p=p))


def agent() -> PolicyAgent:
    """Entry point for ``--opponent examples.train_policy:agent``."""
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(
            f"no trained weights at {WEIGHTS} -- run: python3 examples/train_policy.py"
        )
    return PolicyAgent(Policy.load(WEIGHTS))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=int, default=30000)
    p.add_argument("--opponent", default="random")
    p.add_argument("--size", type=int, default=3)
    p.add_argument("-k", "--k", type=int, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "--entropy",
        type=float,
        default=0.01,
        help="entropy bonus; stops the policy collapsing onto one opening",
    )
    p.add_argument("--eval-games", type=int, default=200)
    args = p.parse_args(argv)
    rows = cols = args.size
    k = args.k or min(rows, cols)

    print(
        "training a %d-unit policy on %dx%d (k=%d) against %s"
        % (args.hidden, rows, cols, k, args.opponent)
    )
    policy = train(
        args.games,
        args.opponent,
        rows,
        cols,
        k,
        args.hidden,
        args.lr,
        args.batch,
        args.seed,
        not args.no_augment,
        args.entropy,
    )
    policy.save(WEIGHTS)
    print(f"saved {WEIGHTS}\n")
    for spec in ("random", "greedy", "perfect"):
        print(
            evaluate(
                policy,
                spec,
                args.eval_games if spec != "perfect" else 40,
                rows,
                cols,
                k,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
