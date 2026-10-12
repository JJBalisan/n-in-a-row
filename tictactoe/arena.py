"""Running games: one at a time, in matches, or in bulk as training data."""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable, Iterator

from tictactoe import agents, game


@dataclasses.dataclass
class GameRecord:
    """One finished game, as written to JSONL by match and selfplay.

    Attributes:
        rows: Board height.
        cols: Board width.
        k: Stones in a row needed to win.
        moves: Cell indices in the order played, X first.
        winner: 0 for a draw, 1 if X won, 2 if O won.
    """

    rows: int
    cols: int
    k: int
    moves: list[int] = dataclasses.field(default_factory=list)
    winner: int | None = None  # 0 draw, 1 X, 2 O

    def result_for(self, player: int) -> float:
        """Returns 1.0, 0.0 or -1.0: the game's result for ``player``."""
        if self.winner == game.DRAW:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def to_json(self) -> str:
        """Returns the record as one line of JSON."""
        return json.dumps(
            {
                "rows": self.rows,
                "cols": self.cols,
                "k": self.k,
                "moves": self.moves,
                "winner": self.winner,
            }
        )


def play_game(
    x_agent: agents.Agent,
    o_agent: agents.Agent,
    rows: int = 3,
    cols: int | None = None,
    k: int | None = None,
    on_move: Callable[[game.Board, int, int], object] | None = None,
) -> GameRecord:
    """Plays one game to the end.

    Args:
        x_agent: Moves first.
        o_agent: Moves second.
        rows: Board height.
        cols: Board width; defaults to ``rows``.
        k: Stones in a row needed to win; defaults to the shorter side.
        on_move: Called as ``on_move(board, mover, move)`` after each move.

    Returns:
        The move list and the result.
    """
    board = game.Board(rows, cols, k)
    players = {game.X: x_agent, game.O: o_agent}
    for a in players.values():
        a.reset(board)
    while not board.is_terminal:
        mover = board.to_move
        move = players[mover].select(board)
        board.push(move)
        for a in players.values():
            a.observe(board, move)
        if on_move is not None:
            on_move(board, mover, move)
    return GameRecord(
        board.rows, board.cols, board.k, board.moves_played(), board.winner
    )


@dataclasses.dataclass
class MatchResult:
    """Running totals for a match, from agent A's point of view.

    Attributes:
        games: Games played.
        wins: Games A won.
        draws: Games drawn.
        losses: Games A lost.
        as_x: A's wins, draws and losses as X, keyed 'w', 'd' and 'l'.
        as_o: The same, as O.
    """

    games: int = 0
    wins: int = 0  # for agent A
    draws: int = 0
    losses: int = 0
    as_x: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"w": 0, "d": 0, "l": 0}
    )
    as_o: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"w": 0, "d": 0, "l": 0}
    )

    @property
    def score(self) -> float:
        """A's chess score per game: a win is 1, a draw is a half."""
        return (
            (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0
        )

    def summary(self, name_a: str, name_b: str) -> str:
        """Formats the totals as two lines of text.

        Args:
            name_a: Agent A's name.
            name_b: Agent B's name.

        Returns:
            W-D-L overall and per colour, plus the score.
        """
        return (
            "%s vs %s over %d games: %d-%d-%d (W-D-L), score %.3f\n"
            "  as X: %d-%d-%d    as O: %d-%d-%d"
            % (
                name_a,
                name_b,
                self.games,
                self.wins,
                self.draws,
                self.losses,
                self.score,
                self.as_x["w"],
                self.as_x["d"],
                self.as_x["l"],
                self.as_o["w"],
                self.as_o["d"],
                self.as_o["l"],
            )
        )


def play_match(
    agent_a: agents.Agent,
    agent_b: agents.Agent,
    games: int = 100,
    rows: int = 3,
    cols: int | None = None,
    k: int | None = None,
    alternate: bool = True,
    records: list[GameRecord] | None = None,
    progress: Callable[[int, int, MatchResult], object] | None = None,
) -> MatchResult:
    """Plays a match, swapping colours each game so first-move edge cancels.

    Args:
        agent_a: The agent the result is scored for; X in even games.
        agent_b: Its opponent.
        games: Number of games.
        rows: Board height.
        cols: Board width; defaults to ``rows``.
        k: Stones in a row needed to win; defaults to the shorter side.
        alternate: Swap colours between games; if False A is always X.
        records: If given, every game's record is appended to it.
        progress: Called as ``progress(done, games, result)`` after each.

    Returns:
        The totals.
    """
    res = MatchResult()
    for i in range(games):
        a_is_x = (i % 2 == 0) or not alternate
        rec = play_game(
            agent_a if a_is_x else agent_b,
            agent_b if a_is_x else agent_a,
            rows,
            cols,
            k,
        )
        if records is not None:
            records.append(rec)
        a_side = game.X if a_is_x else game.O
        outcome = rec.result_for(a_side)
        bucket = res.as_x if a_is_x else res.as_o
        res.games += 1
        if outcome > 0:
            res.wins += 1
            bucket["w"] += 1
        elif outcome < 0:
            res.losses += 1
            bucket["l"] += 1
        else:
            res.draws += 1
            bucket["d"] += 1
        if progress is not None:
            progress(i + 1, games, res)
    return res


def selfplay(
    agent_spec: str,
    games: int,
    rows: int = 3,
    cols: int | None = None,
    k: int | None = None,
    seed: int | None = None,
) -> Iterator[GameRecord]:
    """Streams games between two independently seeded copies of one agent.

    Args:
        agent_spec: An agent spec, as :func:`tictactoe.agents.load_agent`
            reads it.
        games: Number of games.
        rows: Board height.
        cols: Board width; defaults to ``rows``.
        k: Stones in a row needed to win; defaults to the shorter side.
        seed: Seeds both agents of every game; None for fresh randomness.

    Yields:
        One record per game.
    """
    rng = random.Random(seed)
    for _ in range(games):
        a = agents.load_agent(agent_spec, random.Random(rng.getrandbits(32)))
        b = agents.load_agent(agent_spec, random.Random(rng.getrandbits(32)))
        yield play_game(a, b, rows, cols, k)


def replay(record: GameRecord) -> Iterator[tuple[game.Board, int, float]]:
    """Walks a recorded game, yielding ``(board_before_move, move, value)``.

    ``value`` is the final result seen from the side to move -- exactly the
    target a value head wants.  Each yielded board is a fresh copy, so callers
    can encode it at leisure.

    Args:
        record: The game to walk.

    Yields:
        One (board, move, value) triple per move.
    """
    board = game.Board(record.rows, record.cols, record.k)
    for move in record.moves:
        value = record.result_for(board.to_move)
        yield board.copy(), move, value
        board.push(move)
