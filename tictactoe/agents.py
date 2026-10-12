"""Players. Anything with ``select(board) -> int`` is an agent.

That one-method contract is the seam a neural network plugs into: point
``--opponent`` at ``mymodule:factory`` and the CLI, the arena and the GUI will
all use your net without knowing anything about it.
"""

from __future__ import annotations

import importlib
import random
from typing import Callable

from tictactoe import game, search


class Agent:
    """Base class. Subclasses implement :meth:`select`.

    Attributes:
        name: A short label used in match summaries and prompts.
    """

    name = "agent"

    def select(self, board: game.Board) -> int:
        """Chooses a move.

        Args:
            board: The position, with this agent to move. Implementations
                may push and pop moves but must leave it as they found it.

        Returns:
            A legal cell index.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def observe(self, board: game.Board, move: int) -> None:
        """Called after every move by either side. Stateless agents ignore it.

        Args:
            board: The position after the move.
            move: The cell just played.
        """

    def reset(self, board: game.Board) -> None:
        """Called at the start of each game.

        Args:
            board: The empty board the game will be played on.
        """

    def explain(self, board: game.Board) -> str | None:
        """Gives an optional one-line rationale, shown in verbose mode.

        Args:
            board: The position the last move was chosen in.

        Returns:
            The rationale for the last :meth:`select`, or None.
        """
        return None

    def __str__(self) -> str:
        return self.name


class RandomAgent(Agent):
    """Plays uniformly at random over the legal moves."""

    name = "random"

    def __init__(self, rng: random.Random | None = None) -> None:
        """Creates the agent.

        Args:
            rng: Source of randomness; a fresh unseeded one if None.
        """
        self.rng = rng or random.Random()

    def select(self, board: game.Board) -> int:
        """Returns a uniformly random legal move."""
        return self.rng.choice(board.legal_moves())


class GreedyAgent(Agent):
    """Takes a win, blocks a loss, otherwise plays near the centre at random.

    The right first sparring partner for a fresh network: beatable, but it
    punishes anything that never learns to complete or stop a line.
    """

    name = "greedy"

    def __init__(self, rng: random.Random | None = None) -> None:
        """Creates the agent.

        Args:
            rng: Breaks ties; a fresh unseeded one if None.
        """
        self.rng = rng or random.Random()

    def select(self, board: game.Board) -> int:
        """Returns a win, else a block, else a random most-central move."""
        me = board.to_move
        them = game.opponent(me)
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
    """Alpha-beta search. ``depth=None`` solves exactly on small boards.

    Attributes:
        engine: The searcher; it keeps its transposition table between moves.
        epsilon: Probability of playing a random move instead of searching.
        last: The latest search result, or None after a random move.
    """

    def __init__(
        self,
        depth: int | None = None,
        time_limit: float | None = None,
        epsilon: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        """Creates the agent.

        Args:
            depth: Search depth cap, or None to search as deep as allowed.
            time_limit: Seconds per move, or None for no limit.
            epsilon: Probability of a random move instead of a search.
            rng: Source of randomness; a fresh unseeded one if None.
        """
        self.engine = search.Negamax(max_depth=depth, time_limit=time_limit)
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.last: search.SearchResult | None = None
        self.name = "perfect" if depth is None else "negamax:%d" % depth
        if epsilon:
            self.name = f"noisy:{epsilon:g}:{self.name}"

    def select(self, board: game.Board) -> int:
        """Returns the searched best move, or a random one with p=epsilon."""
        if self.epsilon and self.rng.random() < self.epsilon:
            self.last = None
            return self.rng.choice(board.legal_moves())
        result = self.engine.search(board)
        self.last = result
        if result.move is None:
            return self.rng.choice(board.legal_moves())
        return result.move

    def explain(self, board: game.Board) -> str | None:
        """Summarises the last search: outcome, depth, nodes and time."""
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

    def __init__(
        self, fn: Callable[[game.Board], int], name: str = "callable"
    ) -> None:
        """Wraps a function.

        Args:
            fn: Maps a board to a cell index.
            name: The agent's label.
        """
        self.fn = fn
        self.name = name

    def select(self, board: game.Board) -> int:
        """Returns ``fn(board)`` as an int."""
        return int(self.fn(board))


class HumanAgent(Agent):
    """Reads a move from a prompt: 'b2', '2,3', a bare index, or a command.

    Commands are 'undo', 'help' and 'quit', with short forms.
    """

    name = "human"

    def __init__(
        self,
        prompt: str = "your move",
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], object] = print,
    ) -> None:
        """Creates the agent.

        Args:
            prompt: Text shown before the suggested cell.
            input_fn: Reads one line given a prompt.
            output_fn: Shows help and error messages.
        """
        self.prompt = prompt
        self.input_fn = input_fn
        self.output_fn = output_fn

    def select(self, board: game.Board) -> int:
        """Prompts until a readable move is entered.

        Args:
            board: The position; it must have at least one legal move.

        Returns:
            The cell index entered. Whether it is empty is not checked.

        Raises:
            Undo: When the user asks to take back a move.
            SystemExit: When the user quits, or input ends or is
                interrupted.
        """
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
            except game.InvalidMove as exc:
                self.output_fn(f"  {exc}")


class Undo(Exception):  # noqa: N818 (public name)
    """Raised by :class:`HumanAgent` to request a take-back."""


BUILTINS = {"random", "greedy", "perfect", "human"}


def load_agent(spec: str, rng: random.Random | None = None) -> Agent:
    """Turns a command-line string into an agent.

    Built-ins::

        random                 uniform over legal moves
        greedy                 win / block / centre
        perfect                full-width alpha-beta (exact on small boards)
        negamax:6              alpha-beta to depth 6
        time:0.5               alpha-beta with a half-second budget per move
        noisy:0.2:perfect      the given agent, but random 20% of the time
        human                  prompts on stdin

    Anything else is read as ``module.path:attribute``. The attribute may be an
    :class:`Agent`, an Agent subclass, or a zero-argument factory returning one
    (or returning a ``fn(board) -> int``, which is wrapped).

    Args:
        spec: The agent spec.
        rng: Shared by the built-ins; a fresh unseeded one if None.

    Returns:
        A new agent, or the attribute itself when it is an Agent instance.

    Raises:
        ValueError: If a built-in's numeric tail does not parse, or the spec
            is unknown and has no attribute part.
        ImportError: If the module cannot be imported.
        AttributeError: If the module has no such attribute.
        TypeError: If the attribute does not yield an agent.
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
    """Plays a random move with probability epsilon, else defers to another."""

    def __init__(
        self, base: Agent, epsilon: float, rng: random.Random | None = None
    ) -> None:
        """Wraps an agent.

        Args:
            base: The agent to defer to.
            epsilon: Probability of a random move.
            rng: Source of randomness; a fresh unseeded one if None.
        """
        self.base = base
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.name = f"noisy:{epsilon:g}:{base.name}"

    def select(self, board: game.Board) -> int:
        """Returns a random legal move with p=epsilon, else the base's move."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(board.legal_moves())
        return self.base.select(board)

    def explain(self, board: game.Board) -> str | None:
        """Returns the base agent's rationale."""
        return self.base.explain(board)
