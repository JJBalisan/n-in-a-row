"""Players. Anything with ``select(board) -> int`` is an agent.

That one-method contract is the seam a neural network plugs into: point
``--opponent`` at ``mymodule:factory`` and the CLI, the arena and the GUI will
all use your net without knowing anything about it.
"""

from __future__ import annotations

import importlib
import random
from typing import Callable

from .game import Board, InvalidMove, opponent
from .search import Negamax, SearchResult


class Agent:
    """Base class. Subclasses implement :meth:`select`."""

    name = "agent"

    def select(self, board: Board) -> int:
        raise NotImplementedError

    def observe(self, board: Board, move: int) -> None:
        """Called after every move by either side. Stateless agents ignore it."""

    def reset(self, board: Board) -> None:
        """Called at the start of each game."""

    def explain(self, board: Board) -> str | None:
        """Optional one-line rationale, shown by the CLI in verbose mode."""
        return None

    def __str__(self) -> str:
        return self.name


class RandomAgent(Agent):
    name = "random"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def select(self, board: Board) -> int:
        return self.rng.choice(board.legal_moves())


class GreedyAgent(Agent):
    """Takes a win, blocks a loss, otherwise plays near the centre at random.

    The right first sparring partner for a fresh network: beatable, but it
    punishes anything that never learns to complete or stop a line.
    """

    name = "greedy"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def select(self, board: Board) -> int:
        me = board.to_move
        them = opponent(me)
        wins = board.winning_moves(me)
        if wins:
            return self.rng.choice(wins)
        blocks = board.winning_moves(them)
        if blocks:
            return self.rng.choice(blocks)
        moves = board.legal_moves()
        cr, cc = (board.rows - 1) / 2.0, (board.cols - 1) / 2.0
        best = min(
            ((r - cr) ** 2 + (c - cc) ** 2)
            for r, c in (board.coord(m) for m in moves)
        )
        near = [
            m
            for m in moves
            if ((board.coord(m)[0] - cr) ** 2 + (board.coord(m)[1] - cc) ** 2)
            == best
        ]
        return self.rng.choice(near)


class NegamaxAgent(Agent):
    """Alpha-beta search. ``depth=None`` solves exactly on small boards."""

    def __init__(
        self,
        depth: int | None = None,
        time_limit: float | None = None,
        epsilon: float = 0.0,
        rng: random.Random | None = None,
    ):
        self.engine = Negamax(max_depth=depth, time_limit=time_limit)
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.last: SearchResult | None = None
        self.name = "perfect" if depth is None else "negamax:%d" % depth
        if epsilon:
            self.name = f"noisy:{epsilon:g}:{self.name}"

    def select(self, board: Board) -> int:
        if self.epsilon and self.rng.random() < self.epsilon:
            self.last = None
            return self.rng.choice(board.legal_moves())
        result = self.engine.search(board)
        self.last = result
        if result.move is None:
            return self.rng.choice(board.legal_moves())
        return result.move

    def explain(self, board: Board) -> str | None:
        r = self.last
        if r is None:
            return "played at random (epsilon)"
        return "%s, depth %d, %d nodes, %.2fs%s" % (
            r.outcome,
            r.depth,
            r.nodes,
            r.seconds,
            "" if r.exact else " (horizon-limited)",
        )


class FunctionAgent(Agent):
    """Wraps a plain ``fn(board) -> int``. Handy for a quick network policy."""

    def __init__(self, fn: Callable[[Board], int], name: str = "callable"):
        self.fn = fn
        self.name = name

    def select(self, board: Board) -> int:
        return int(self.fn(board))


class HumanAgent(Agent):
    """Reads a move from a prompt. Understands 'b2', '2,3', a bare index, and 'undo'."""

    name = "human"

    def __init__(
        self, prompt: str = "your move", input_fn=input, output_fn=print
    ):
        self.prompt = prompt
        self.input_fn = input_fn
        self.output_fn = output_fn

    def select(self, board: Board) -> int:
        while True:
            try:
                raw = self.input_fn(
                    f"{self.prompt} ({board.label(board.legal_moves()[0])}) > "
                )
            except (EOFError, KeyboardInterrupt):
                raise SystemExit("\nstopped")
            cmd = raw.strip().lower()
            if cmd in ("q", "quit", "exit", "resign"):
                raise SystemExit("resigned")
            if cmd in ("u", "undo"):
                raise Undo()
            if cmd in ("?", "h", "help"):
                self.output_fn(
                    "enter a cell like b2, or 2,3, or a bare index; "
                    "'undo' takes back a move, 'quit' leaves"
                )
                continue
            try:
                return board.parse_move(raw)
            except InvalidMove as exc:
                self.output_fn(f"  {exc}")


class Undo(Exception):
    """Raised by :class:`HumanAgent` to request a take-back."""


BUILTINS = {"random", "greedy", "perfect", "human"}


def load_agent(spec: str, rng: random.Random | None = None) -> Agent:
    """Turn a command-line string into an agent.

    Built-ins::

        random                 uniform over legal moves
        greedy                 win / block / centre
        perfect                full-width alpha-beta (exact on small boards)
        negamax:6              alpha-beta to depth 6
        time:0.5               alpha-beta with a half-second budget per move
        noisy:0.2:perfect      the given agent, but random 20% of the time
        human                  prompts on stdin

    Anything else is read as ``module.path:attribute``. The attribute may be an
    :class:`Agent`, an Agent subclass, or a zero-argument factory returning one.
    """
    rng = rng or random.Random()
    spec = spec.strip()
    head, _, tail = spec.partition(":")
    head = head.strip()

    if head == "noisy":
        eps_s, _, rest = tail.partition(":")
        base = load_agent(rest or "perfect", rng)
        if isinstance(base, NegamaxAgent):
            base.epsilon = float(eps_s)
            base.name = f"noisy:{eps_s}:{base.name}"
            return base
        return NoisyAgent(base, float(eps_s), rng)
    if head == "random":
        return RandomAgent(rng)
    if head == "greedy":
        return GreedyAgent(rng)
    if head == "perfect":
        return NegamaxAgent(depth=None, rng=rng)
    if head == "negamax":
        return NegamaxAgent(depth=int(tail), rng=rng)
    if head == "time":
        return NegamaxAgent(depth=None, time_limit=float(tail), rng=rng)
    if head == "human":
        return HumanAgent()

    if not tail:
        raise ValueError(
            "unknown agent {!r} -- expected one of {}, negamax:D, time:S, "
            "noisy:P:AGENT, or module:attribute".format(
                spec, ", ".join(sorted(BUILTINS))
            )
        )
    module = importlib.import_module(head)
    obj = getattr(module, tail)
    if isinstance(obj, Agent):
        return obj
    if isinstance(obj, type) and issubclass(obj, Agent):
        return obj()
    if callable(obj):
        made = obj()
        if isinstance(made, Agent):
            return made
        if callable(made):
            return FunctionAgent(made, name=spec)
    raise TypeError(f"{spec} did not yield an Agent")


class NoisyAgent(Agent):
    def __init__(
        self, base: Agent, epsilon: float, rng: random.Random | None = None
    ):
        self.base = base
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.name = f"noisy:{epsilon:g}:{base.name}"

    def select(self, board: Board) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(board.legal_moves())
        return self.base.select(board)

    def explain(self, board: Board):
        return self.base.explain(board)
