"""Extra checks for the s2 solver: how module:attribute specs get an rng.

Stdlib only:  python3 -m unittest discover -s tests
"""

import os
import random
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import agents

SEEN = []


class TakesRng(agents.Agent):
    name = "takes-rng"

    def __init__(self, rng=None):
        self.rng = rng

    def select(self, board):
        return board.legal_moves()[0]


def rng_factory(rng=None):
    SEEN.append(rng)
    return TakesRng(rng)


def plain_factory():
    return TakesRng()


def rng_class_factory():
    return TakesRng


class TestRngReachesFactories(unittest.TestCase):
    def test_factory_with_rng_parameter_gets_the_loaders_rng(self):
        rng = random.Random(5)
        agent = agents.load_agent(f"{__name__}:rng_factory", rng)
        self.assertIs(SEEN[-1], rng)
        self.assertIs(agent.rng, rng)

    def test_zero_argument_factory_still_works(self):
        agent = agents.load_agent(f"{__name__}:plain_factory", random.Random(1))
        self.assertIsNone(agent.rng)

    def test_class_with_rng_parameter_gets_it(self):
        rng = random.Random(6)
        agent = agents.load_agent(f"{__name__}:TakesRng", rng)
        self.assertIs(agent.rng, rng)

    def test_class_returned_by_factory_gets_it(self):
        rng = random.Random(7)
        agent = agents.load_agent(f"{__name__}:rng_class_factory", rng)
        self.assertIs(agent.rng, rng)


if __name__ == "__main__":
    unittest.main()
