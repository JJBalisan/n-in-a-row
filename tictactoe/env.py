"""Reinforcement-learning wrapper. The only module that needs NumPy.

Observations are *canonical*: plane 0 is always the stones of the player to
move, so a network never has to learn "am I X or O".  Rewards are returned from
the perspective of whoever just moved.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from tictactoe import agents, game

#: (own stones, opponent stones, side-to-move) -- the third plane is constant
#: over the board and matters on sizes where moving first is a real edge.
N_PLANES = 3


def encode(
    board: game.Board, dtype: np.typing.DTypeLike = np.float32
) -> np.ndarray:
    """Returns the canonical observation of a position.

    Args:
        board: The position.
        dtype: Element type of the result.

    Returns:
        An array of shape ``(3, rows, cols)``: the side to move's stones, the
        opponent's stones, and a constant plane that is 1 when X is to move.
    """
    cells = np.asarray(board.cells, dtype=np.int8).reshape(
        board.rows, board.cols
    )
    me = board.to_move
    obs = np.zeros((N_PLANES, board.rows, board.cols), dtype=dtype)
    obs[0] = cells == me
    obs[1] = cells == game.opponent(me)
    obs[2] = 1.0 if me == game.X else 0.0
    return obs


def legal_mask(board: game.Board) -> np.ndarray:
    """Returns a boolean array of length ``rows*cols``, True where legal.

    Mask your policy logits with this.

    Args:
        board: The position.

    Returns:
        The legal-move mask.
    """
    return np.asarray(board.legal_mask(), dtype=bool)


def dihedral(
    obs: np.ndarray, policy: np.ndarray, rows: int, cols: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns every board symmetry of an example, for data augmentation.

    There are 8 on a square board (the full dihedral group) and 4 on a
    rectangle, where transposing would change the shape.

    Args:
        obs: An observation of shape ``(planes, rows, cols)``.
        policy: Per-cell values of length ``rows*cols``.
        rows: Board height.
        cols: Board width.

    Returns:
        ``(obs, policy)`` pairs, the identity first, each transformed alike.
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
            variants.append(
                (np.ascontiguousarray(oo), np.ascontiguousarray(pp).reshape(-1))
            )
    return variants


class TicTacToeEnv:
    """Single-game environment with the usual reset/step shape.

    Two modes:

    * ``opponent=None`` -- **self-play**. Every call to :meth:`step` places one
      stone and hands the turn over; the observation always belongs to whoever
      is to move next. Use this for AlphaZero-style training.
    * ``opponent=<Agent>`` -- the built-in opponent replies inside :meth:`step`,
      so the agent under training only ever sees its own turns.

    Attributes:
        board: The current position.
        opponent: The built-in opponent, or None for self-play.
        agent_plays: The side the trained agent plays when there is an
            opponent.
        illegal_move: 'raise' or 'lose'.
        n_actions: Number of cells.
        observation_shape: ``(3, rows, cols)``.
    """

    def __init__(
        self,
        rows: int = 3,
        cols: int | None = None,
        k: int | None = None,
        opponent: agents.Agent | None = None,
        agent_plays: int | None = game.X,
        illegal_move: str = "raise",
        seed: int | None = None,
    ) -> None:
        """Creates the environment with an empty board.

        Args:
            rows: Board height.
            cols: Board width; defaults to ``rows``.
            k: Stones in a row needed to win; defaults to the shorter side.
            opponent: Replies inside :meth:`step`; None for self-play.
            agent_plays: The trained agent's side when there is an opponent.
            illegal_move: 'raise' to raise InvalidMove, or 'lose' to end the
                game with reward -1.
            seed: Seeds the environment's ``rng``.

        Raises:
            ValueError: If ``illegal_move`` is neither 'raise' nor 'lose'.
        """
        if illegal_move not in ("raise", "lose"):
            raise ValueError("illegal_move must be 'raise' or 'lose'")
        self.rows = rows
        self.cols = rows if cols is None else cols
        self.k = min(self.rows, self.cols) if k is None else k
        self.opponent = opponent
        self.agent_plays = agent_plays
        self.illegal_move = illegal_move
        self.rng = random.Random(seed)
        self.board = game.Board(self.rows, self.cols, self.k)
        self.n_actions = self.board.size
        self.observation_shape = (N_PLANES, self.rows, self.cols)

    # ---- gym-ish API -----------------------------------------------------

    def reset(self, swap_sides: bool = False) -> np.ndarray:
        """Starts a new game.

        With an opponent, the opponent moves first when it holds X.

        Args:
            swap_sides: With an opponent, switch the trained agent's side.

        Returns:
            The first observation for the side to move.
        """
        self.board = game.Board(self.rows, self.cols, self.k)
        if swap_sides and self.opponent is not None:
            self.agent_plays = game.opponent(self.agent_plays)
        if self.opponent is not None:
            self.opponent.reset(self.board)
            if self.board.to_move != self.agent_plays:
                self._opponent_move()
        return self.observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Plays a move, and the opponent's reply if there is one.

        Args:
            action: The cell index to play.

        Returns:
            ``(obs, reward, done, info)``. Reward is from the perspective of
            the side that just moved: ``+1`` win, ``-1`` loss (including an
            illegal move under 'lose'), ``0`` otherwise. ``info`` holds
            to_move, ply, winner and legal_mask, or illegal and error.

        Raises:
            InvalidMove: If the move is illegal and ``illegal_move`` is
                'raise'.
        """
        mover = self.board.to_move
        try:
            self.board.push(int(action))
        except game.InvalidMove as exc:
            if self.illegal_move == "raise":
                raise
            return (
                self.observation(),
                -1.0,
                True,
                {"illegal": True, "error": str(exc)},
            )

        if self.board.is_terminal:
            return self.observation(), self._reward(mover), True, self._info()

        if self.opponent is not None:
            self._opponent_move()
            if self.board.is_terminal:
                return (
                    self.observation(),
                    self._reward(mover),
                    True,
                    self._info(),
                )

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
        """Returns the canonical observation of the current position."""
        return encode(self.board)

    def legal_mask(self) -> np.ndarray:
        """Returns the current legal-move mask."""
        return legal_mask(self.board)

    def legal_actions(self) -> list[int]:
        """Returns the current legal cell indices."""
        return self.board.legal_moves()

    def symmetries(
        self, obs: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns :func:`dihedral` for this board's shape."""
        return dihedral(obs, policy, self.rows, self.cols)

    def render(self, color: bool = False) -> str:
        """Returns the board as text; see :meth:`Board.render`."""
        return self.board.render(color=color)

    def clone(self) -> TicTacToeEnv:
        """Returns a copy with its own board and the same opponent object."""
        env = TicTacToeEnv(
            self.rows,
            self.cols,
            self.k,
            self.opponent,
            self.agent_plays,
            self.illegal_move,
        )
        env.board = self.board.copy()
        return env


class BatchEnv:
    """A fixed-width bank of games stepped together.

    Neural nets want batches; running 512 boards side by side turns one forward
    pass into 512 moves.  Finished games auto-reset: :meth:`step` reports the
    reward and done flag of the move that ended a game, alongside the first
    observation of the next one.

    Attributes:
        envs: The games.
        n: Number of games.
        n_actions: Number of cells.
        observation_shape: ``(3, rows, cols)``.
    """

    def __init__(
        self,
        n: int,
        rows: int = 3,
        cols: int | None = None,
        k: int | None = None,
        opponent_spec: str | None = None,
        seed: int | None = None,
        illegal_move: str = "raise",
    ) -> None:
        """Creates ``n`` independently seeded games.

        Args:
            n: Number of games.
            rows: Board height.
            cols: Board width; defaults to ``rows``.
            k: Stones in a row needed to win; defaults to the shorter side.
            opponent_spec: Agent spec for a per-game opponent; None for
                self-play.
            seed: Seeds every game and opponent.
            illegal_move: 'raise' or 'lose', as in :class:`TicTacToeEnv`.
        """
        rng = random.Random(seed)
        self.envs = [
            TicTacToeEnv(
                rows,
                cols,
                k,
                opponent=None
                if opponent_spec is None
                else agents.load_agent(
                    opponent_spec, random.Random(rng.getrandbits(32))
                ),
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
        """Resets every game; returns observations stacked on axis 0."""
        return np.stack([e.reset() for e in self.envs])

    def step(
        self, actions: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Steps every game, resetting the ones that finish.

        Every second game to finish swaps sides on reset (with an opponent).

        Args:
            actions: One cell index per game.

        Returns:
            Stacked observations, float32 rewards, bool dones and a list of
            info dicts, one entry per game.
        """
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
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def legal_masks(self) -> np.ndarray:
        """Returns every game's legal-move mask, stacked on axis 0."""
        return np.stack([e.legal_mask() for e in self.envs])
