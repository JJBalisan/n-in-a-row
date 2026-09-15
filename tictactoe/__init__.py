"""Generalised tic-tac-toe: engine, search, agents, RL environment.

    from tictactoe import Board, load_agent, play_game
    board = Board(5, 5, k=4)          # 5x5, four in a row
"""

from .agents import Agent, FunctionAgent, GreedyAgent, NegamaxAgent, RandomAgent, load_agent
from .arena import GameRecord, MatchResult, play_game, play_match, replay, selfplay
from .game import DRAW, EMPTY, Board, InvalidMove, O, X, opponent
from .search import Negamax, SearchResult, solve

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
