"""Tests for load_agent's module:attribute path, HumanAgent and NoisyAgent.

Stdlib only:  python3 -m unittest discover -s tests
"""

import os
import random
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))
sys.path.insert(0, _TESTS)  # so load_agent can import fake_agent

import fake_agent

from tictactoe.agents import (
    FunctionAgent,
    HumanAgent,
    NegamaxAgent,
    NoisyAgent,
    Undo,
    load_agent,
)
from tictactoe.game import Board


class TestLoadAgentModuleAttribute(unittest.TestCase):
    """The module:attribute spec accepts four shapes and rejects the rest."""

    def test_instance_is_returned_as_is(self):
        self.assertIs(load_agent("fake_agent:INSTANCE"), fake_agent.INSTANCE)

    def test_subclass_is_instantiated(self):
        agent = load_agent("fake_agent:FirstLegal")
        self.assertIsInstance(agent, fake_agent.FirstLegal)
        self.assertEqual(agent.select(Board.from_moves([0, 1])), 2)

    def test_factory_returning_an_agent(self):
        agent = load_agent("fake_agent:make_agent")
        self.assertIsInstance(agent, fake_agent.LastLegal)
        self.assertIsNot(agent, fake_agent.INSTANCE)

    def test_factory_returning_a_callable_is_wrapped(self):
        agent = load_agent("fake_agent:make_policy")
        self.assertIsInstance(agent, FunctionAgent)
        self.assertEqual(agent.name, "fake_agent:make_policy")
        self.assertEqual(agent.select(Board.from_moves([8])), 7)

    def test_non_agent_attribute_raises_type_error(self):
        for spec in ("fake_agent:NOT_AN_AGENT", "fake_agent:make_nothing"):
            with self.subTest(spec=spec), self.assertRaises(TypeError):
                load_agent(spec)

    def test_missing_attribute_and_module(self):
        with self.assertRaises(AttributeError):
            load_agent("fake_agent:no_such_thing")
        with self.assertRaises(ModuleNotFoundError):
            load_agent("no_such_module_xyz:agent")


class _Script:
    """Feeds HumanAgent canned input and records everything it prints."""

    def __init__(self, *lines):
        self.lines = list(lines)
        self.prompts = []
        self.output = []

    def input_fn(self, prompt):
        self.prompts.append(prompt)
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)

    def output_fn(self, text):
        self.output.append(text)

    def agent(self):
        return HumanAgent(input_fn=self.input_fn, output_fn=self.output_fn)


class TestHumanAgent(unittest.TestCase):
    """HumanAgent driven through its input_fn/output_fn seams."""

    def test_valid_move_forms(self):
        for text, expected in [("b2", 4), ("B3", 7), ("2,3", 5), ("8", 8),
                               ("  c1  ", 2)]:
            with self.subTest(text=text):
                script = _Script(text)
                self.assertEqual(script.agent().select(Board()), expected)
                self.assertEqual(script.output, [])

    def test_help_prints_usage_then_keeps_asking(self):
        script = _Script("?", "help", "a1")
        self.assertEqual(script.agent().select(Board()), 0)
        self.assertEqual(len(script.output), 2)
        self.assertIn("undo", script.output[0])
        self.assertEqual(len(script.prompts), 3)

    def test_invalid_input_reports_and_reprompts(self):
        script = _Script("z9", "nonsense", "", "99", "c3")
        self.assertEqual(script.agent().select(Board()), 8)
        self.assertEqual(len(script.output), 4)
        self.assertIn("off the board", script.output[0])
        self.assertIn("could not read", script.output[1])
        self.assertIn("empty input", script.output[2])
        self.assertIn("off the board", script.output[3])
        self.assertEqual(len(script.prompts), 5)

    @unittest.expectedFailure
    def test_occupied_cell_is_reprompted(self):
        """KNOWN-FAIL (new, not in the audit): HumanAgent returns taken cells.

        parse_move checks only the board bounds, so select() hands back an
        occupied cell; cmd_play then pushes it, InvalidMove escapes to main()
        and a typo ends the game with exit code 2 instead of a re-prompt.
        """
        board = Board.from_moves([4])
        script = _Script("b2", "a1")
        self.assertEqual(script.agent().select(board), 0)

    def test_undo_raises_undo(self):
        for text in ("u", "undo", " UNDO "):
            with self.subTest(text=text), self.assertRaises(Undo):
                _Script(text).agent().select(Board())

    def test_quit_resigns(self):
        for text in ("q", "quit", "exit", "resign"):
            with self.subTest(text=text):
                with self.assertRaises(SystemExit) as ctx:
                    _Script(text).agent().select(Board())
                self.assertEqual(str(ctx.exception), "resigned")

    def test_end_of_input_stops(self):
        with self.assertRaises(SystemExit) as ctx:
            _Script().agent().select(Board())
        self.assertIn("stopped", str(ctx.exception))

    def test_prompt_shows_an_example_legal_cell(self):
        script = _Script("c3")
        script.agent().select(Board.from_moves([0]))
        self.assertIn("(b1)", script.prompts[0])


class _Refuses(fake_agent.LastLegal):
    """A base agent that fails the test if NoisyAgent ever consults it."""

    def select(self, board):
        raise AssertionError("base agent was consulted")


class TestNoisyAgent(unittest.TestCase):
    """NoisyAgent defers to its base with probability 1 - epsilon."""

    def test_epsilon_zero_always_plays_the_base_move(self):
        agent = NoisyAgent(fake_agent.LastLegal(), 0.0, random.Random(0))
        rng = random.Random(1)
        for _ in range(50):
            board = Board(4, 4, 4)       # no one can win within 6 plies
            for _ in range(rng.randrange(7)):
                board.push(rng.choice(board.legal_moves()))
            self.assertEqual(agent.select(board), board.legal_moves()[-1])

    def test_epsilon_one_never_consults_the_base(self):
        agent = NoisyAgent(_Refuses(), 1.0, random.Random(0))
        board = Board(4, 4, 3)
        picks = set()
        for _ in range(200):
            move = agent.select(board)
            self.assertIn(move, board.legal_moves())
            picks.add(move)
        self.assertGreater(len(picks), 8, "random moves are not spread out")

    def test_name_and_explain_delegate(self):
        agent = NoisyAgent(fake_agent.LastLegal(), 0.25)
        self.assertEqual(agent.name, "noisy:0.25:last-legal")
        self.assertIsNone(agent.explain(Board()))

    def test_load_agent_wraps_non_negamax_bases(self):
        agent = load_agent("noisy:1:fake_agent:FirstLegal", random.Random(0))
        self.assertIsInstance(agent, NoisyAgent)
        self.assertEqual(agent.epsilon, 1.0)
        self.assertIsInstance(agent.base, fake_agent.FirstLegal)

    def test_load_agent_folds_epsilon_into_negamax(self):
        agent = load_agent("noisy:0.3:negamax:2", random.Random(0))
        self.assertIsInstance(agent, NegamaxAgent)
        self.assertEqual(agent.epsilon, 0.3)
        self.assertEqual(agent.name, "noisy:0.3:negamax:2")

    def test_negamax_epsilon_one_plays_at_random(self):
        agent = load_agent("noisy:1:negamax:1", random.Random(0))
        board = Board.from_moves([0, 3, 1, 4])     # X wins at c1
        picks = {agent.select(board) for _ in range(60)}
        self.assertGreater(len(picks), 1)
        self.assertEqual(agent.explain(board), "played at random (epsilon)")

    def test_negamax_epsilon_zero_takes_the_win(self):
        agent = load_agent("noisy:0:negamax:2", random.Random(0))
        board = Board.from_moves([0, 3, 1, 4])
        for _ in range(10):
            self.assertEqual(agent.select(board), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
