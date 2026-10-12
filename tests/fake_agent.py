"""Targets for load_agent's module:attribute specs, used by test_agents.py.

Each attribute is one of the shapes load_agent accepts (or must reject):

    fake_agent:INSTANCE       an Agent instance, returned as is
    fake_agent:FirstLegal     an Agent subclass, instantiated with no arguments
    fake_agent:make_agent     a zero-argument factory returning an Agent
    fake_agent:make_policy    a zero-argument factory returning fn(board) -> int
    fake_agent:NOT_AN_AGENT   neither callable nor an Agent: TypeError
    fake_agent:make_nothing   a factory returning a non-callable: TypeError
    fake_agent:Interrupts     an agent whose select() acts like Ctrl-C
"""

from tictactoe.agents import Agent


class FirstLegal(Agent):
    """Plays the lowest-numbered empty cell."""

    name = "first-legal"

    def select(self, board):
        """Returns the first legal move.

        Args:
            board: The position to move in.

        Returns:
            The lowest legal cell index.
        """
        return board.legal_moves()[0]


class LastLegal(Agent):
    """Plays the highest-numbered empty cell."""

    name = "last-legal"

    def select(self, board):
        """Returns the last legal move.

        Args:
            board: The position to move in.

        Returns:
            The highest legal cell index.
        """
        return board.legal_moves()[-1]


class Interrupts(Agent):
    """Raises KeyboardInterrupt on its first move, as if Ctrl-C was pressed."""

    name = "interrupts"

    def select(self, board):
        """Raises KeyboardInterrupt.

        Args:
            board: The position to move in (ignored).

        Raises:
            KeyboardInterrupt: Always.
        """
        raise KeyboardInterrupt


INSTANCE = LastLegal()

NOT_AN_AGENT = "just a string"


def make_agent():
    """Builds an agent.

    Returns:
        A fresh LastLegal agent.
    """
    return LastLegal()


def make_policy():
    """Builds a bare policy function, which load_agent wraps.

    Returns:
        A callable mapping a board to its highest legal cell index.
    """
    return lambda board: board.legal_moves()[-1]


def make_nothing():
    """Builds something that is not an agent.

    Returns:
        An int, which load_agent must reject.
    """
    return 42
