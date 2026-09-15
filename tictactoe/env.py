"""Reinforcement-learning wrapper. The only module that needs NumPy.

Observations are *canonical*: plane 0 is always the stones of the player to
move, so a network never has to learn "am I X or O".  Rewards are returned from
the perspective of whoever just moved.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from .agents import Agent, load_agent
from .game import Board, InvalidMove, X, opponent

#: (own stones, opponent stones, side-to-move) -- the third plane is constant
#: over the board and matters on sizes where moving first is a real edge.
N_PLANES = 3


def encode(board: Board, dtype=np.float32) -> np.ndarray:
    """Canonical observation of shape ``(3, rows, cols)``."""
    cells = np.asarray(board.cells, dtype=np.int8).reshape(board.rows, board.cols)
    me = board.to_move
    obs = np.zeros((N_PLANES, board.rows, board.cols), dtype=dtype)
    obs[0] = cells == me
    obs[1] = cells == opponent(me)
    obs[2] = 1.0 if me == X else 0.0
    return obs


def legal_mask(board: Board) -> np.ndarray:
    """Boolean array of length ``rows*cols``. Mask your policy logits with this."""
    return np.asarray(board.legal_mask(), dtype=bool)


def dihedral(obs: np.ndarray, policy: np.ndarray, rows: int, cols: int):
    """All board symmetries, for data augmentation.

    Yields ``(obs, policy)`` pairs: 8 of them on a square board (the full
    dihedral group), 4 on a rectangle where transposing would change the shape.
    """
    pol = policy.reshape(rows, cols)
    square = rows == cols
    variants = []
    for flip in (False, True):
        o = obs[:, :, ::-1] if flip else obs
        p = pol[:, ::-1] if flip else pol
        for rot in range(4):
            if rot and not square and rot % 2:
                continue
            oo = np.rot90(o, rot, axes=(1, 2))
            pp = np.rot90(p, rot)
            variants.append((np.ascontiguousarray(oo), np.ascontiguousarray(pp).reshape(-1)))
    return variants


class TicTacToeEnv:
    """Single-game environment with the usual reset/step shape.

    Two modes:

    * ``opponent=None`` -- **self-play**. Every call to :meth:`step` places one
      stone and hands the turn over; the observation always belongs to whoever
      is to move next. Use this for AlphaZero-style training.
    * ``opponent=<Agent>`` -- the built-in opponent replies inside :meth:`step`,
      so the agent under training only ever sees its own turns.
    """

    def __init__(self, rows: int = 3, cols: int | None = None, k: int | None = None,
                 opponent: Agent | None = None, agent_plays: int | None = X,
                 illegal_move: str = "raise", seed: int | None = None):
        if illegal_move not in ("raise", "lose"):
            raise ValueError("illegal_move must be 'raise' or 'lose'")
        self.rows = rows
        self.cols = rows if cols is None else cols
        self.k = min(self.rows, self.cols) if k is None else k
        self.opponent = opponent
        self.agent_plays = agent_plays
        self.illegal_move = illegal_move
        self.rng = random.Random(seed)
        self.board = Board(self.rows, self.cols, self.k)
        self.n_actions = self.board.size
        self.observation_shape = (N_PLANES, self.rows, self.cols)

    # ---- gym-ish API -----------------------------------------------------

    def reset(self, swap_sides: bool = False) -> np.ndarray:
        """Start a new game. ``swap_sides`` alternates who moves first."""
        self.board = Board(self.rows, self.cols, self.k)
        if swap_sides and self.opponent is not None:
            self.agent_plays = opponent(self.agent_plays)
        if self.opponent is not None:
            self.opponent.reset(self.board)
            if self.board.to_move != self.agent_plays:
                self._opponent_move()
        return self.observation()

    def step(self, action: int):
        """Returns ``(obs, reward, done, info)``.

        Reward is from the perspective of the side that just moved: ``+1`` win,
        ``-1`` loss, ``0`` otherwise.
        """
        mover = self.board.to_move
        try:
            self.board.push(int(action))
        except InvalidMove as exc:
            if self.illegal_move == "raise":
                raise
            return self.observation(), -1.0, True, {"illegal": True, "error": str(exc)}

        if self.board.is_terminal:
            return self.observation(), self._reward(mover), True, self._info()

        if self.opponent is not None:
            self._opponent_move()
            if self.board.is_terminal:
                return self.observation(), self._reward(mover), True, self._info()

        return self.observation(), 0.0, False, self._info()

    def _opponent_move(self) -> None:
        move = self.opponent.select(self.board)
        self.board.push(move)

    def _reward(self, mover: int) -> float:
        r = self.board.result_for(mover)
        return 0.0 if r is None else float(r)

    def _info(self) -> dict:
        return {
            "to_move": self.board.to_move,
            "ply": self.board.ply,
            "winner": self.board.winner,
            "legal_mask": self.legal_mask(),
        }

    # ---- observation helpers --------------------------------------------

    def observation(self) -> np.ndarray:
        return encode(self.board)

    def legal_mask(self) -> np.ndarray:
        return legal_mask(self.board)

    def legal_actions(self) -> list[int]:
        return self.board.legal_moves()

    def symmetries(self, obs: np.ndarray, policy: np.ndarray):
        return dihedral(obs, policy, self.rows, self.cols)

    def render(self, color: bool = False) -> str:
        return self.board.render(color=color)

    def clone(self) -> TicTacToeEnv:
        env = TicTacToeEnv(self.rows, self.cols, self.k, self.opponent,
                           self.agent_plays, self.illegal_move)
        env.board = self.board.copy()
        return env


class BatchEnv:
    """A fixed-width bank of games stepped together.

    Neural nets want batches; running 512 boards side by side turns one forward
    pass into 512 moves.  Finished games auto-reset, and ``last_rewards`` /
    ``last_dones`` report what happened on the step that ended them.
    """

    def __init__(self, n: int, rows: int = 3, cols: int | None = None,
                 k: int | None = None, opponent_spec: str | None = None,
                 seed: int | None = None, illegal_move: str = "raise"):
        rng = random.Random(seed)
        self.envs = [
            TicTacToeEnv(
                rows, cols, k,
                opponent=None if opponent_spec is None else load_agent(
                    opponent_spec, random.Random(rng.getrandbits(32))),
                illegal_move=illegal_move,
                seed=rng.getrandbits(32),
            )
            for _ in range(n)
        ]
        self.n = n
        self.observation_shape = self.envs[0].observation_shape
        self.n_actions = self.envs[0].n_actions
        self._alternate = 0

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: Sequence[int]):
        obs, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, d, i = env.step(int(a))
            if d:
                self._alternate += 1
                o = env.reset(swap_sides=self._alternate % 2 == 0)
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(i)
        return (np.stack(obs), np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)

    def legal_masks(self) -> np.ndarray:
        return np.stack([e.legal_mask() for e in self.envs])
