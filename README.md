# n-in-a-row

Generalised n-in-a-row: any board size, any run length, one engine underneath a
terminal front end, a windowed front end, and a reinforcement-learning
environment.

Three layers, deliberately separated:

| layer | module | needs |
|---|---|---|
| engine, search, agents, arena | `n_in_a_row.game`, `.search`, `.agents`, `.arena` | nothing — standard library only |
| terminal + window | `n_in_a_row.cli`, `n_in_a_row.gui` | the above (plus `tkinter` for the GUI) |
| RL environment | `n_in_a_row.env` | NumPy |

The GUI and the CLI both call `agent.select(board)` and nothing else, so
anything you train drops into either one without changes.

## Quick start

```bash
cd n-in-a-row
python3 -m n_in_a_row play --size 5 -k 4 --opponent negamax:5   # play in the terminal
python3 -m n_in_a_row.gui --size 5 -k 4 --opponent negamax:5    # play in a window
python3 -m unittest discover -s tests                           # stdlib only; see Tests
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
from n_in_a_row import Agent


class MyNet(Agent):
    name = "mynet"

    def select(self, board) -> int:
        obs = encode(board)  # (3, rows, cols) float32
        logits = self.net(obs)  # your forward pass
        logits[~np.asarray(board.legal_mask())] = -np.inf
        return int(logits.argmax())


def agent():  # the factory the CLI imports
    return MyNet()
```

Then everything works:

```bash
python3 -m n_in_a_row match --a mymodule:agent --b greedy --games 400
python3 -m n_in_a_row.gui --opponent mymodule:agent
```

`module:attribute` may resolve to an `Agent` instance, an `Agent` subclass, or a
zero-argument factory returning either.

## The RL environment

```python
from n_in_a_row.env import TicTacToeEnv
from n_in_a_row.agents import load_agent

env = TicTacToeEnv(
    rows=5,
    cols=5,
    k=4,
    opponent=load_agent("noisy:0.3:perfect"),
    illegal_move="lose",  # or "raise"
)
obs = env.reset()  # (3, 5, 5) float32
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

Throughput on this machine (`python3 -m n_in_a_row bench`): ~100k engine
moves/s, ~25k batched env moves/s, ~20k search nodes/s on a 4×4.

## Training data from the engine

```bash
python3 -m n_in_a_row selfplay --games 20000 --agent noisy:0.25:perfect --out data.jsonl
```

One JSON game per line. `n_in_a_row.replay(record)` turns each into
`(board, move, value)` triples where `value` is the final result from the side
to move — the target a value head wants.

## Worked examples

Both are NumPy-only so they run with nothing installed, and both end by
plugging themselves back into the CLI.

**`examples/learn_from_solver.py`** — distils the search engine into a small
MLP by asking it for the optimal moves in every position. On 3×3 it reaches
perfect play in about 25 seconds:

```text
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
python3 -m n_in_a_row match --a examples.learn_from_solver:agent --b perfect --games 100
python3 -m n_in_a_row.gui --opponent examples.learn_from_solver:agent
```

## The search engine

Negamax with alpha-beta, a Zobrist-keyed transposition table, iterative
deepening and move ordering (immediate wins, then blocks, then cells next to
existing stones, then centre). Mate scores count plies from the current
position, so `solve` reports "win in 2" meaning two of your own moves from here.

```bash
python3 -m n_in_a_row solve --size 3               # every opening draws
python3 -m n_in_a_row solve --size 3 a1 a2 b1      # O to move: loss in 2
```

Exact search is capped at 14 empty cells (`SOFT_EXACT_LIMIT`); beyond that it
searches to that depth and flags the result as horizon-limited rather than
hanging. Every `SearchResult` carries `.exact`, so nothing has to guess whether
a value is game-theoretic or a heuristic estimate.

## Layout

```text
n_in_a_row/
  game.py      Board: shape, moves, undo stack, Zobrist hash, win detection, text I/O
  search.py    Negamax: alpha-beta, TT, iterative deepening, threat heuristic
  agents.py    Agent protocol, built-in agents, load_agent() spec parser
  arena.py     play_game, play_match, selfplay, replay
  env.py       TicTacToeEnv, BatchEnv, encode, legal_mask, dihedral   [NumPy]
  cli.py       play / match / selfplay / solve / bench
  gui.py       Tkinter board                                          [tkinter]
examples/      learn_from_solver.py, train_policy.py
tests/         test_engine.py, test_cli.py, test_agents.py (+ fake_agent.py) -- stdlib unittest
```

## Tests

```bash
python3 -m unittest discover -s tests            # anywhere, nothing installed
pip install -e '.[test]'                         # adds NumPy for the env tests
CI=1 python3 -m unittest discover -s tests       # what CI runs
```

The `TestEnv` cases need NumPy. Locally they skip when it is missing; when the
`CI` environment variable is set (GitLab and GitHub set `CI=true`) a missing
NumPy fails them instead, so a green pipeline always means `env.py` was tested.
The `test` extra in `pyproject.toml` installs it. GitHub Actions
(`.github/workflows/ci.yml`) checks formatting, reports ruff findings
without failing yet, then installs the test extra and runs the suite.

Tests marked `@unittest.expectedFailure` pin known, not-yet-fixed defects (their
docstrings say which); the run stays green while they fail and goes red once a
fix makes one pass, so remove the decorator with the fix.

## Releasing

Every release is an annotated tag `vX.Y.Z` on `main`, recorded in
[`CHANGELOG.md`](CHANGELOG.md); `n-in-a-row --version` (and
`n-in-a-row-gui --version`) prints what git says about the checkout it
runs from: `v1.0.0-0-g<sha>` on a release, `v1.0.0-3-g<sha>` three commits
past it, `-dirty` with local edits.

Each MAJOR version also has a codename, the same for all of its MINOR and
PATCH releases: a county of Kenya, a new one per MAJOR. It goes in the
`.0.0` tag's message and every release title on GitHub (`v1.0.0 — Lamu`),
never in the version number.

| MAJOR | Codename |
| --- | --- |
| 1 | Lamu |

1. Between releases, each change adds `changes/<number>-<slug>.md`
   (`python3 lib/changelog.py new <slug>`; see
   [`changes/README.md`](changes/README.md)) rather than editing
   `[Unreleased]`.
2. On a branch, `python3 lib/changelog.py release X.Y.Z` folds the
   fragments into a new `[X.Y.Z] - date` section and rewrites the compare
   links; set the same version in `pyproject.toml` and run
   `/opt/homebrew/bin/uv lock`. Open that as the release PR.
3. Merge to `main`, pull, then tag the merge commit, push the tag and
   publish the GitHub release:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z — <codename>" && git push origin vX.Y.Z
   gh release create vX.Y.Z --title "vX.Y.Z — <codename>" --notes-from-tag
   ```

MAJOR when the public API breaks (`game`, `search`, `agents`, `arena`,
`env`, an agent spec or a CLI flag), MINOR when it grows, PATCH for a fix.

## Interpreters on this machine

- `/usr/bin/python3` and Anaconda's `python3` have `tkinter` and NumPy — the
  GUI and the RL layer both work.
- Homebrew's `python3` (3.14, arm64) has neither out of the box:
  `brew install python-tk` for the GUI, `pip install numpy` for `n_in_a_row.env`.
- The CLI, the engine and the tests need none of that and run on any of them.
