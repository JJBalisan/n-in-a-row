"""Selector checks for #33, #45, #46, #51, written from the issue text.

Stdlib only:  python3 -m unittest discover -s tests
"""

import os
import random
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents, arena, game
from n_in_a_row.harness import zoo


class MyAgent(agents.Agent):
    """Plays the first legal move."""

    name = "mine"

    def select(self, board):
        return board.legal_moves()[0]


class Recorder(MyAgent):
    """Counts resets and observed moves."""

    name = "recorder"

    def __init__(self):
        self.resets = 0
        self.observed = []

    def reset(self, board):
        self.resets += 1

    def observe(self, board, move):
        self.observed.append(move)


class Explains(MyAgent):
    """Explains every move it chose with a fixed string."""

    def explain(self, board):
        return "BASE"


# ---- module attributes reachable from specs ---------------------------

RECORDERS = []


def recorder_factory():
    made = Recorder()
    RECORDERS.append(made)
    return made


def zero_arg_factory():
    return MyAgent()


def zero_arg_function_factory():
    return lambda board: board.legal_moves()[-1]


SEEN = {}


def defaulted_param_factory(size=3):
    SEEN["size"] = size
    return MyAgent()


def rng_factory(rng=None):
    SEEN["rng"] = rng
    return MyAgent()


def class_factory():
    return MyAgent


SHARED_NOISY = agents.NegamaxAgent(depth=2, epsilon=0.3, rng=random.Random(0))


def _spec(attribute):
    return f"{__name__}:{attribute}"


def _forced_win_misses(spec, seed=0, n=1000):
    # X to move wins at cell 2; five legal moves, so a random move misses
    # 4/5 of the time.
    agent = agents.load_agent(spec, random.Random(seed))
    board = game.Board.from_moves([0, 3, 1, 4])
    return sum(agent.select(board) != 2 for _ in range(n))


class TestFactoryContractKept(unittest.TestCase):
    """#46 / #51 must not break the documented zero-argument factory."""

    def test_zero_arg_factory_still_loads(self):
        agent = agents.load_agent(_spec("zero_arg_factory"), random.Random(1))
        self.assertIsInstance(agent, MyAgent)

    def test_zero_arg_function_factory_still_wrapped(self):
        agent = agents.load_agent(
            _spec("zero_arg_function_factory"), random.Random(1)
        )
        self.assertIsInstance(agent, agents.FunctionAgent)
        self.assertEqual(agent.select(game.Board()), 8)

    def test_unrelated_defaulted_parameter_is_not_fed_the_rng(self):
        SEEN.clear()
        agents.load_agent(_spec("defaulted_param_factory"), random.Random(1))
        self.assertEqual(SEEN["size"], 3)

    def test_rng_parameter_receives_the_given_rng(self):
        SEEN.clear()
        rng = random.Random(5)
        agents.load_agent(_spec("rng_factory"), rng)
        self.assertIs(SEEN["rng"], rng)

    def test_zoo_factories_still_callable_with_no_arguments(self):
        for name in ("saboteur", "stealth", "sandbagger", "locked", "noisy"):
            with self.subTest(factory=name):
                made = getattr(zoo, name)()
                self.assertIsInstance(made, agents.Agent)
                arena.play_game(made, agents.RandomAgent(random.Random(0)))


class TestFactoryReturningClass(unittest.TestCase):
    """#46."""

    def test_each_load_gets_a_fresh_instance_that_plays(self):
        a = agents.load_agent(_spec("class_factory"), random.Random(0))
        b = agents.load_agent(_spec("class_factory"), random.Random(0))
        self.assertIsInstance(a, MyAgent)
        self.assertIsNot(a, b)
        record = arena.play_game(a, agents.RandomAgent(random.Random(0)))
        self.assertTrue(record.moves)


class TestSeededZooSpecs(unittest.TestCase):
    """#51 through wrappers and the arena."""

    def _games(self, spec, seed):
        agent = agents.load_agent(spec, random.Random(seed))
        opponent = agents.RandomAgent(random.Random(42))
        return tuple(
            tuple(arena.play_game(agent, opponent).moves) for _ in range(8)
        )

    def test_noisy_wrapped_zoo_factory_is_reproducible(self):
        spec = "noisy:0.3:n_in_a_row.harness.zoo:saboteur"
        self.assertEqual(self._games(spec, 7), self._games(spec, 7))

    def test_zoo_saboteur_as_o_is_reproducible(self):
        spec = "n_in_a_row.harness.zoo:stealth"

        def run():
            agent = agents.load_agent(spec, random.Random(3))
            x = agents.RandomAgent(random.Random(11))
            return [tuple(arena.play_game(x, agent).moves) for _ in range(8)]

        self.assertEqual(run(), run())


class TestNoisyFolding(unittest.TestCase):
    """#45."""

    def test_builtin_negamax_still_folds(self):
        agent = agents.load_agent("noisy:0.25:negamax:3", random.Random(0))
        self.assertIs(type(agent), agents.NegamaxAgent)
        self.assertEqual(agent.epsilon, 0.25)

    def test_outer_zero_does_not_erase_inner_noise(self):
        misses = _forced_win_misses("noisy:0:noisy:0.5:perfect")
        # 0.5 * 4/5 = 0.4 expected.
        self.assertGreater(misses, 300)
        self.assertLess(misses, 500)

    def test_shared_instance_with_own_epsilon_is_untouched(self):
        name = SHARED_NOISY.name
        agents.load_agent(f"noisy:0.9:{_spec('SHARED_NOISY')}")
        self.assertEqual(SHARED_NOISY.epsilon, 0.3)
        self.assertEqual(SHARED_NOISY.name, name)

    def test_shared_instance_noise_compounds_not_replaced(self):
        # Shared epsilon 0.3 kept; outer 0.5 on top: 1 - 0.5 * 0.7 = 0.65
        # random, misses ~0.52. Overwriting with 0.5 gives ~0.40.
        misses = _forced_win_misses(f"noisy:0.5:{_spec('SHARED_NOISY')}")
        self.assertGreater(misses, 460)
        self.assertLess(misses, 600)


class TestNoisyAgentHooks(unittest.TestCase):
    """#33."""

    def test_nested_noise_resets_the_inner_agent_once_per_game(self):
        RECORDERS.clear()
        agent = agents.load_agent(
            f"noisy:0.2:noisy:0.3:{_spec('recorder_factory')}",
            random.Random(0),
        )
        for _ in range(3):
            record = arena.play_game(
                agent, agents.RandomAgent(random.Random(1))
            )
        (inner,) = RECORDERS
        self.assertEqual(inner.resets, 3)
        self.assertEqual(inner.observed[-len(record.moves) :], record.moves)

    def test_explain_follows_the_last_select(self):
        agent = agents.NoisyAgent(Explains(), 0.5, random.Random(0))
        board = game.Board()
        base_seen = random_seen = False
        for _ in range(60):
            move = agent.select(board)
            text = agent.explain(board)
            if text == "BASE":
                base_seen = True
            else:
                random_seen = True
            self.assertIn(move, board.legal_moves())
        self.assertTrue(base_seen and random_seen)

    def test_explain_after_reset_does_not_raise(self):
        agent = agents.NoisyAgent(Explains(), 1.0, random.Random(0))
        board = game.Board()
        agent.select(board)
        agent.reset(board)
        agent.explain(board)

    def test_a_new_game_does_not_inherit_the_random_flag(self):
        # Differential (s1 differs): after reset no move has been played
        # in this game, so explain must not say the last move was random.
        agent = agents.NoisyAgent(Explains(), 1.0, random.Random(0))
        board = game.Board()
        agent.select(board)
        agent.reset(board)
        self.assertEqual(agent.explain(board), "BASE")


if __name__ == "__main__":
    unittest.main()
