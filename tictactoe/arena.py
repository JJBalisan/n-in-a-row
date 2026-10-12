"""Running games: one at a time, in matches, or in bulk to make training data."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from .agents import Agent, load_agent
from .game import DRAW, Board, O, X


@dataclass
class GameRecord:
    rows: int
    cols: int
    k: int
    moves: list[int] = field(default_factory=list)
    winner: int | None = None  # 0 draw, 1 X, 2 O

    def result_for(self, player: int) -> float:
        if self.winner == DRAW:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def to_json(self) -> str:
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
    x_agent: Agent,
    o_agent: Agent,
    rows: int = 3,
    cols: int | None = None,
    k: int | None = None,
    on_move=None,
) -> GameRecord:
    """Play one game to the end and return the move list plus the result."""
    board = Board(rows, cols, k)
    players = {X: x_agent, O: o_agent}
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


@dataclass
class MatchResult:
    games: int = 0
    wins: int = 0  # for agent A
    draws: int = 0
    losses: int = 0
    as_x: dict[str, int] = field(
        default_factory=lambda: {"w": 0, "d": 0, "l": 0}
    )
    as_o: dict[str, int] = field(
        default_factory=lambda: {"w": 0, "d": 0, "l": 0}
    )

    @property
    def score(self) -> float:
        """Chess scoring: a win is 1, a draw is a half."""
        return (
            (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0
        )

    def summary(self, name_a: str, name_b: str) -> str:
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
    agent_a: Agent,
    agent_b: Agent,
    games: int = 100,
    rows: int = 3,
    cols: int | None = None,
    k: int | None = None,
    alternate: bool = True,
    records: list[GameRecord] | None = None,
    progress=None,
) -> MatchResult:
    """Play ``games`` games, swapping colours each time so first-move advantage cancels."""
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
        a_side = X if a_is_x else O
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
    """Stream games between two independently-seeded copies of one agent."""
    rng = random.Random(seed)
    for _ in range(games):
        a = load_agent(agent_spec, random.Random(rng.getrandbits(32)))
        b = load_agent(agent_spec, random.Random(rng.getrandbits(32)))
        yield play_game(a, b, rows, cols, k)


def replay(record: GameRecord):
    """Walk a recorded game, yielding ``(board_before_move, move, value)``.

    ``value`` is the final result seen from the side to move -- exactly the
    target a value head wants.  Each yielded board is a fresh copy, so callers
    can encode it at leisure.
    """
    board = Board(record.rows, record.cols, record.k)
    for move in record.moves:
        value = record.result_for(board.to_move)
        yield board.copy(), move, value
        board.push(move)
