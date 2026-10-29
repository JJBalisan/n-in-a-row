"""Generalised n-in-a-row: engine, search, agents, RL environment.

The package re-exports the names most callers need.

Example:
    from n_in_a_row import Board, load_agent, play_game
    board = Board(5, 5, k=4)          # 5x5, four in a row
"""

from n_in_a_row import agents, arena, game, search

# Public re-exports: ``from n_in_a_row import Board`` and friends (see README).
Agent = agents.Agent
FunctionAgent = agents.FunctionAgent
GreedyAgent = agents.GreedyAgent
NegamaxAgent = agents.NegamaxAgent
RandomAgent = agents.RandomAgent
load_agent = agents.load_agent

GameRecord = arena.GameRecord
MatchResult = arena.MatchResult
play_game = arena.play_game
play_match = arena.play_match
replay = arena.replay
selfplay = arena.selfplay

DRAW = game.DRAW
EMPTY = game.EMPTY
Board = game.Board
InvalidMove = game.InvalidMove
O = game.O  # noqa: E741 (the player constant; public name)
X = game.X
opponent = game.opponent

Negamax = search.Negamax
SearchResult = search.SearchResult
solve = search.solve

__version__ = "1.0.0"

__all__ = [
    "DRAW",
    "EMPTY",
    "Agent",
    "Board",
    "FunctionAgent",
    "GameRecord",
    "GreedyAgent",
    "InvalidMove",
    "MatchResult",
    "Negamax",
    "NegamaxAgent",
    "O",
    "RandomAgent",
    "SearchResult",
    "X",
    "__version__",
    "load_agent",
    "opponent",
    "play_game",
    "play_match",
    "replay",
    "selfplay",
    "solve",
]
