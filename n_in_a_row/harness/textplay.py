"""Text play: a position as a prompt, a reply as a move, for language models.

:func:`render_prompt` draws the board with ``Board.render()`` and asks for a
final ``MOVE: <cell>`` line. :func:`parse_reply` reads a reply back and
says what went wrong when it cannot: ``unparseable`` (no single move in
it), ``off_board`` (a cell that does not exist) or ``occupied`` (a taken
cell). :class:`TextAgent` wraps any ``prompt -> reply`` callable as an
agent, so a language model plugs into the arena, the monitors and the zoo
unchanged. Nothing here calls a network; the callable is the caller's.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple

from n_in_a_row import agents, game

#: Reply statuses, in the order a reply is checked.
OK, UNPARSEABLE, OFF_BOARD, OCCUPIED = (
    "ok",
    "unparseable",
    "off_board",
    "occupied",
)

_MOVE_LINE = re.compile(r"move\s*:\s*([^\n]*)", re.IGNORECASE)
_CELL_WORD = re.compile(r"\b[a-z]+\d+\b")
_CELL_SHAPE = re.compile(r"[a-z]+\d+|\d+[,;x]\d+|\d+")
_TRIM = " \t.,;:!?`*'\"()[]"


class ReplyResult(NamedTuple):
    """What a reply said, read against a position.

    Attributes:
        status: ``ok``, ``unparseable``, ``off_board`` or ``occupied``.
        move: The cell index named, or None if no cell on the board was.
        text: The part of the reply read as the move.
        detail: A one-line reason, fit to show the model on a retry.
    """

    status: str
    move: int | None
    text: str
    detail: str

    @property
    def ok(self) -> bool:
        """True if the reply named a legal move."""
        return self.status == OK


class TextTurn(NamedTuple):
    """One exchange with the model.

    Attributes:
        ply: Stones on the board when it was asked.
        prompt: What the model was sent.
        reply: What it answered.
        result: How the answer was read.
    """

    ply: int
    prompt: str
    reply: str
    result: ReplyResult


class ReplyError(Exception):
    """Raised by :class:`TextAgent` when no reply named a legal move.

    Attributes:
        result: How the last reply was read.
    """

    def __init__(self, result: ReplyResult) -> None:
        super().__init__(result.detail)
        self.result = result


def render_prompt(board: game.Board) -> str:
    """Describes a position and asks for a move.

    Args:
        board: A position with at least one legal move.

    Returns:
        The rules, the board, the legal moves and the reply format.
    """
    me = game.SYMBOL[board.to_move]
    them = game.SYMBOL[game.opponent(board.to_move)]
    legal = ", ".join(board.label(m) for m in board.legal_moves())
    return (
        f"Board: {board.rows}x{board.cols}, {board.k} in a row wins.\n"
        f"You play {me}; your opponent plays {them}. It is your move.\n"
        "Columns are letters, rows are numbers; '.' is an empty cell.\n"
        f"\n{board.render()}\n\n"
        f"Legal moves: {legal}\n"
        "End your reply with one line of the form MOVE: <cell>, "
        "for example MOVE: b2.\n"
    )


def _extract(reply: str, strict: bool) -> str | None:
    """Finds the part of a reply that names the move, or None."""
    if strict:
        return reply.strip()
    lines = _MOVE_LINE.findall(reply)
    if lines:
        return lines[-1].strip(_TRIM)
    whole = reply.strip().strip(_TRIM)
    if _CELL_SHAPE.fullmatch(whole.lower().replace(" ", "")):
        return whole
    words = set(_CELL_WORD.findall(reply.lower()))
    if len(words) == 1:
        return words.pop()
    return None


def parse_reply(
    board: game.Board, reply: str, *, strict: bool = False
) -> ReplyResult:
    """Reads a model's reply as a move in a position.

    The move is taken from the last ``MOVE:`` line if there is one, else
    from the whole reply if it is just a cell, else from the one cell name
    (such as ``b2``) the prose mentions; two different cell names are
    unparseable. Cells are read with ``Board.parse_move``, so ``b2``,
    ``2,2`` and the bare index ``4`` all work.

    Args:
        board: The position the reply is about.
        reply: The model's text.
        strict: Read only the whole reply, stripped, as a cell.

    Returns:
        The reading; ``status`` is ``ok`` only for a legal move.
    """
    text = _extract(reply, strict)
    if not text:
        return ReplyResult(UNPARSEABLE, None, "", "no move found in the reply")
    try:
        move = board.parse_move(text)
    except game.InvalidMove as exc:
        shaped = _CELL_SHAPE.fullmatch(text.lower().replace(" ", ""))
        status = OFF_BOARD if shaped else UNPARSEABLE
        return ReplyResult(status, None, text, str(exc))
    if move not in board.legal_moves():
        return ReplyResult(
            OCCUPIED, move, text, f"{board.label(move)} is already taken"
        )
    return ReplyResult(OK, move, text, "")


class TextAgent(agents.Agent):
    """An agent that plays by sending prompts to a ``prompt -> reply`` callable.

    Every exchange is kept in :attr:`turns`. A failed reply is retried with
    the reason appended to the prompt, up to ``retries`` times; after that
    the agent raises :class:`ReplyError`, or plays ``fallback``'s move if
    one is given. Failures are counted, never repaired silently.

    Attributes:
        model: The callable.
        retries: Extra attempts after a failed reply.
        fallback: Plays when every attempt failed, or None to raise.
        strict: Passed to :func:`parse_reply`.
        turns: Every exchange so far.
        failures: Replies that did not name a legal move.
        fallbacks: Moves the fallback agent played.
    """

    def __init__(
        self,
        model: Callable[[str], str],
        *,
        retries: int = 0,
        fallback: agents.Agent | None = None,
        strict: bool = False,
        name: str = "text",
    ) -> None:
        """Wraps a model.

        Args:
            model: Maps a prompt to a reply.
            retries: Extra attempts after a failed reply.
            fallback: Plays when every attempt failed; None raises.
            strict: Read only whole replies as cells.
            name: The agent's label.
        """
        self.model = model
        self.retries = retries
        self.fallback = fallback
        self.strict = strict
        self.name = name
        self.turns: list[TextTurn] = []
        self.failures = 0
        self.fallbacks = 0

    def select(self, board: game.Board) -> int:
        """Asks the model for a move, retrying on a bad reply.

        Args:
            board: The position, with at least one legal move.

        Returns:
            The legal move the model named, or the fallback's move.

        Raises:
            ReplyError: If no attempt named a legal move and there is no
                fallback.
        """
        base = render_prompt(board)
        prompt = base
        for _ in range(self.retries + 1):
            reply = self.model(prompt)
            result = parse_reply(board, reply, strict=self.strict)
            self.turns.append(TextTurn(board.ply, prompt, reply, result))
            if result.ok and result.move is not None:
                return result.move
            self.failures += 1
            prompt = (
                f"{base}\nYour last reply was not a legal move "
                f"({result.status}: {result.detail}). Try again.\n"
            )
        if self.fallback is None:
            raise ReplyError(result)
        self.fallbacks += 1
        return self.fallback.select(board)
