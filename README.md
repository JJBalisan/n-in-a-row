# tictactoe

Generalised n-in-a-row: any board size, any run length, one engine underneath a
terminal front end, a windowed front end, and a reinforcement-learning
environment.

Three layers, deliberately separated:

| layer | module | needs |
|---|---|---|
| engine, search, agents, arena | `tictactoe.game`, `.search`, `.agents`, `.arena` | nothing — standard library only |
| terminal + window | `tictactoe.cli`, `tictactoe.gui` | the above (plus `tkinter` for the GUI) |
| RL environment | `tictactoe.env` | NumPy |

The GUI and the CLI both call `agent.select(board)` and nothing else, so
anything you train drops into either one without changes.

## Quick start

```bash
cd tictactoe
python3 -m tictactoe play --size 5 -k 4 --opponent negamax:5   # play in the terminal
python3 -m tictactoe.gui --size 5 -k 4 --opponent negamax:5    # play in a window
python3 -m unittest discover -s tests                          # 28 tests, no deps
```

Board shape is `--size N` for N×N, or `--rows R --cols C` for a rectangle;
`-k K` sets how many in a row wins (default: the shorter side). `--size 3` is
classic tic-tac-toe, `--rows 6 --cols 7 -k 4` is Connect Four without gravity.

## Agents

Everywhere an agent is named — `--opponent`, `--a`, `--b` — you can write:

| spec | plays |
|---|---|
| `random` | uniformly over legal moves |
| `greedy` | takes a win, blocks a loss, else plays centrally |
| `perfect` | alpha-beta, exact on small boards, horizon-limited above 14 empty cells |
| `negamax:6` | alpha-beta to depth 6 |
| `time:0.5` | alpha-beta with a half-second budget per move |
| `noisy:0.2:perfect` | any of the above, but random 20% of the time |
| `human` | prompts on stdin |
| `mymodule:factory` | **yours** |

`noisy:` is the useful one for training: a perfect opponent that never loses
gives a fresh network no gradient to climb, whereas `noisy:0.3:perfect`
punishes real mistakes while still losing when it should.

## Plugging in a network

Implement one method:

```python
from tictactoe import Agent

class MyNet(Agent):
    name = "mynet"
    def select(self, board) -> int:
        obs = encode(board)                 # (3, rows, cols) float32
        logits = self.net(obs)              # your forward pass
        logits[~np.asarray(board.legal_mask())] = -np.inf
        return int(logits.argmax())

def agent():                                # the factory the CLI imports
    return MyNet()
```

Then everything works:

```bash
python3 -m tictactoe match --a mymodule:agent --b greedy --games 400
python3 -m tictactoe.gui --opponent mymodule:agent
```

`module:attribute` may resolve to an `Agent` instance, an `Agent` subclass, or a
zero-argument factory returning either.

## The RL environment

```python
from tictactoe.env import TicTacToeEnv
from tictactoe.agents import load_agent

env = TicTacToeEnv(rows=5, cols=5, k=4,
                   opponent=load_agent("noisy:0.3:perfect"),
                   illegal_move="lose")     # or "raise"
obs = env.reset()                           # (3, 5, 5) float32
obs, reward, done, info = env.step(action)  # reward is the mover's: +1 / 0 / -1
```

- **Observations are canonical.** Plane 0 is always the stones of the player to
  move, plane 1 the opponent's, plane 2 a constant marking which colour that is.
  One network plays both sides; it never has to learn "am I X or O".
- **Mask your logits** with `env.legal_mask()` — a bool array of length
  `rows*cols`. Or set `illegal_move="lose"` and let the agent learn the rules
  the hard way.
- **`opponent=None` is self-play**: every `step` places one stone and hands over
  the turn, which is the shape AlphaZero-style training wants.
- **`env.symmetries(obs, policy)`** returns all 8 dihedral transforms of a
  square board (4 on a rectangle) with the policy transformed to match — eight
  times the data for free.
- **`BatchEnv(n, ...)`** runs a bank of games in lockstep so one forward pass
  becomes `n` moves. Finished games reset themselves and alternate colours.

Throughput on this machine (`python3 -m tictactoe bench`): ~100k engine
moves/s, ~25k batched env moves/s, ~20k search nodes/s on a 4×4.

## Training data from the engine

```bash
python3 -m tictactoe selfplay --games 20000 --agent noisy:0.25:perfect --out data.jsonl
```

One JSON game per line. `tictactoe.replay(record)` turns each into
`(board, move, value)` triples where `value` is the final result from the side
to move — the target a value head wants.

## Worked examples

Both are NumPy-only so they run with nothing installed, and both end by
plugging themselves back into the CLI.

**`examples/learn_from_solver.py`** — distils the search engine into a small
MLP by asking it for the optimal moves in every position. On 3×3 it reaches
perfect play in about 25 seconds:

```
distilled vs random  over 200 games: 187-13-0  (W-D-L), score 0.968
distilled vs greedy  over 200 games:   0-200-0 (W-D-L), score 0.500
distilled vs perfect over  50 games:   0-50-0  (W-D-L), score 0.500
```

Never loses a game. This is the better template to port to PyTorch — the
pipeline is the part worth copying, not the two-layer MLP.

**`examples/train_policy.py`** — plain REINFORCE against a live opponent, for
when you want the reinforcement-learning loop rather than a teacher. It is
honestly weak (it beats `random` comfortably and stalls around 0.25 against
`greedy` after 40k games); vanilla policy gradient with no value baseline and no
search is simply not a strong method here. Included because it shows the
`reset`/`step`/mask/augment loop in about 80 lines.

```bash
python3 examples/learn_from_solver.py --size 3
python3 -m tictactoe match --a examples.learn_from_solver:agent --b perfect --games 100
python3 -m tictactoe.gui --opponent examples.learn_from_solver:agent
```

## The search engine

Negamax with alpha-beta, a Zobrist-keyed transposition table, iterative
deepening and move ordering (immediate wins, then blocks, then cells next to
existing stones, then centre). Mate scores count plies from the current
position, so `solve` reports "win in 2" meaning two of your own moves from here.

```bash
python3 -m tictactoe solve --size 3               # every opening draws
python3 -m tictactoe solve --size 3 a1 a2 b1      # O to move: loss in 2
```

Exact search is capped at 14 empty cells (`SOFT_EXACT_LIMIT`); beyond that it
searches to that depth and flags the result as horizon-limited rather than
hanging. Every `SearchResult` carries `.exact`, so nothing has to guess whether
a value is game-theoretic or a heuristic estimate.

## Layout

```
tictactoe/
  game.py      Board: shape, moves, undo stack, Zobrist hash, win detection, text I/O
  search.py    Negamax: alpha-beta, TT, iterative deepening, threat heuristic
  agents.py    Agent protocol, built-in agents, load_agent() spec parser
  arena.py     play_game, play_match, selfplay, replay
  env.py       TicTacToeEnv, BatchEnv, encode, legal_mask, dihedral   [NumPy]
  cli.py       play / match / selfplay / solve / bench
  gui.py       Tkinter board                                          [tkinter]
examples/      learn_from_solver.py, train_policy.py
tests/         test_engine.py -- 28 tests, stdlib unittest
```

## Interpreters on this machine

- `/usr/bin/python3` and Anaconda's `python3` have `tkinter` and NumPy — the
  GUI and the RL layer both work.
- Homebrew's `python3` (3.14, arm64) has neither out of the box:
  `brew install python-tk` for the GUI, `pip install numpy` for `tictactoe.env`.
- The CLI, the engine and the tests need none of that and run on any of them.
