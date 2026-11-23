# Changelog

Every release is an annotated git tag `vX.Y.Z` on `main`, and every MAJOR
version also has a codename; see "Releasing" in the README.
`n-in-a-row --version` prints the tag or commit it runs from, so a report
can always be matched to an entry here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions: [Semantic Versioning](https://semver.org/spec/v2.0.0.html), where
a MAJOR change breaks the public API (`game`, `search`, `agents`, `arena`,
`env`, an agent spec or a CLI flag), MINOR adds to it, PATCH fixes it.

## [Unreleased]

## [1.3.0] - 2026-11-23

### Added

- `agents.HumanQuit`, also importable as `arena.HumanQuit`: the
  `SystemExit` subclass a `HumanAgent` raises when its player quits or
  input ends. Only it is turned into exit 0; any other agent's
  `SystemExit` exits as before (#38).

### Changed

- A human quit raises `agents.HumanQuit`, a `SystemExit` subclass, in
  place of a plain `SystemExit` (#38).

### Fixed

- `play --opponent human`: the opponent's undo takes back like the
  player's, where it was a traceback, and its quit or end of input
  ends the game with exit 0, where it was exit 1. In `match` and
  `selfplay` a human seat's undo exits 2 with a message, where it was
  a traceback, and its quit or end of input exits 0, where it was
  exit 1. An `Undo` from any seat that is not itself a `HumanAgent`
  exits 2 with the same message, in `play` too (#38).
- One agent object in both seats gets `reset` and `observe` once per
  game, not twice (#44).
- `GameRecord.from_json` reads back what `to_json` wrote (#49).
- `bench` prints the k actually played, not `None` (#53).
- `selfplay` builds the two agents of a built-in spec once and reseeds
  them per game, and a search that runs to the end of the game keeps
  its table from game to game. A time-limited or depth-capped search
  starts each game with an empty table, and `module:attribute` agents
  are still built per game, so the records for a seed are as before
  (#57).
- `Saboteur.throw_games` and `Sandbagger.deviation_games` record the game
  of each `throws` / `deviations` entry, so entries from different games
  can be told apart. The game is the agent's new `games` attribute, which
  each `reset()` raises by one from 0: entries logged before the first
  `reset()` get 0, and with one `reset()` per game the first game is 1
  (#52).
- The env tests `test_n_env_alternate`, `test_n_env_opponent_observe` and
  `test_n_env_select` skip without NumPy (and fail when `CI` is set)
  instead of erroring at import (#83).

## [1.2.0] - 2026-11-22

### Changed

- `module:factory` specs: a factory or class with an `rng` parameter is
  called with the loader's `rng`, and the zoo factories (`saboteur`,
  `stealth`, `sandbagger`, `locked`, `noisy`) take it, so a seeded run
  plays the same games. Zero-argument factories work as before (#51).

### Fixed

- `Negamax.search` reports a won position as a loss for the side to move,
  not a draw (#31).
- `Negamax` rejects a `max_depth` below 1 when it is built, so `negamax:0`
  no longer plays as a search that never looks ahead (#39).
- A search cut by the time limit keeps a win its last pass already
  proved, instead of falling back to an earlier, weaker pass (#62).
- `NoisyAgent` forwards `reset` and `observe` to the agent it wraps and no
  longer explains a random move with the base agent's rationale (#33).
- `noisy:P:AGENT` no longer edits the agent a `module:attribute` spec
  resolves to, and folds P into the epsilon only of a negamax agent it
  built itself from `perfect`, `negamax:D` or `time:S`, so `noisy:` over a
  `module:attribute` or over another `noisy:` wraps and compounds (#45).
- A `module:factory` spec whose factory returns an `Agent` subclass now
  instantiates it instead of failing on the first move (#46).
- The GUI's solver overlay shows a read only on the game, board shape and
  `k` it was computed for, so a late read no longer tags another board
  (#34).
- The GUI keeps the engine's explanation of its move in the detail line
  until the position changes (#50).
- The GUI refuses a size spinbox text over 12 characters before Tcl
  parses it, instead of stalling on a huge number (#63).
- `BatchEnv` swaps each game's side every time that game finishes,
  instead of sharing one counter across games that finish in the same
  step (#36).
- `TicTacToeEnv` calls `opponent.observe` after every move by either
  side, as `arena.play_game` does (part of #42; the unused `seed`
  argument is still open).
- `solver_targets(depth=N)` labels every position with a fresh search, so a
  depth-limited label no longer depends on which positions were labelled
  before it (#37).
- `PolicyAgent` raises a `ValueError` naming the board when its weights were
  trained for a different board size, instead of NumPy's matmul error (#47).
- `TextAgent` passes `reset` and `observe` on to its fallback agent, so a
  stateful fallback sees every game and move (#40).
- `n_in_a_row.__version__` matches the version in `pyproject.toml`
  instead of reporting 1.0.0 (#54).

## [1.1.4] - 2026-11-20

### Fixed

- A search cut short by a time limit keeps its last completed depth, so
  the engine no longer plays a losing move and calls it exact (#30).
- The k >= 8 heuristic stays out of the mate band, so it no longer
  reports false forced wins (#30).
- The GUI drops engine moves computed for a game that has been replaced,
  including across the "Cannot load opponent" dialog (#30).
- New game checks the size controls first: bad input shows a dialog and
  leaves the current game and its search alone (#30).
- GUI engine workers get their board and agent up front, and games that
  share an agent no longer search on it at once (#30).
- Under `CI=1` the GUI tests fail, rather than skip, without tkinter or a
  display; hosted CI runs them under xvfb with uv's own Python (#30).
- A search stopped by any error or Ctrl-C, not only its time limit, now
  takes back its root move, so `search`, `evaluate_moves` and the labeller
  leave the caller's board as they found it (#32).

## [1.1.3] - 2026-11-03

### Changed

- `labels.ExactLabeller` reuses one engine across positions instead of a
  fresh one each, labelling all 4,520 tic-tac-toe positions about 2.7
  times faster with identical labels; the workaround was no longer needed
  after the v1.0.2 engine fix (#17).
- The engine and harness tests share one brute-force minimax reference,
  `tests/brute_force.py`, and enumerate positions with `labels.reachable`
  (#20).

## [1.1.2] - 2026-11-02

### Fixed

- The README and the `zoo` docstrings describe the stealth saboteur as it
  behaves since v1.1.1: it throws blunders the monitor does not flag,
  scored at or under the threshold (#16).

## [1.1.1] - 2026-11-02

### Fixed

- `textplay.parse_reply` reads the first cell on a `MOVE:` line, so
  `MOVE: b2 (centre)` is b2 rather than unparseable (#15).
- The stealth saboteur throws every blunder the monitor does not flag,
  including those scored exactly at the threshold (#16).
- `TextAgent` rejects a negative `retries` with `ValueError` instead of
  failing with `UnboundLocalError` (#18).
- The label cache is stamped with `labels.CACHE_VERSION` and a file with
  another stamp raises `CacheVersionError` (#19).
- `Sandbagger` passes `reset` and `observe` on to its fallback agent (#21).
- The `MOVE:` line must start a word, so `remove:` is not read as a move
  (#22).
- `stats.power` returns 1.0 or 0.0 when both rates are 0 or 1 instead of
  dividing by zero (#23).

## [1.1.0] - 2026-11-02

### Added

- `n_in_a_row.harness`, a shared harness for AI-safety experiments with the
  solver as ground truth: an exact move labeller with an SQLite cache,
  depth-d search monitors, regret metrics, tie sets and covert capacity,
  AUROC / precision / power / CUSUM statistics, saboteur, sandbagger and
  password-locked agents, and text play for language models (#11).
- `n_in_a_row.harness.trainer`: the `learn_from_solver.py` distillation
  pipeline as a library, with pluggable teachers; both examples now use it
  and behave as before (#11).

## [1.0.2] - 2026-11-02

### Fixed

- A `Negamax` reused across positions reports the fastest mate again: a
  mate stored by a deeper search, but beyond the current iteration's
  horizon, could end iterative deepening on a slower mate (20 of 4,520
  tic-tac-toe positions were a few plies off). Outcomes were always
  right (#12).

## [1.0.1] - 2026-11-02

### Changed

- CI runs `bash tests/ci.sh` alone: the format, lint and test steps after
  it ran the same checks a second time (dev-toolchain#49) (#9).

## [1.0.0] - 2026-10-29

### Added

- Engine for any board shape and run length (`--size N`, `--rows R --cols
  C`, `-k K`), standard library only: `n_in_a_row.game`.
- Alpha-beta negamax search with a transposition table, and agents built on
  it: `random`, `greedy`, `perfect`, `negamax:D`, `time:S`, `noisy:P:AGENT`,
  `human`, or any `module:attribute`.
- `arena` matches and the CLI subcommands `play`, `match`, `selfplay`,
  `solve` and `bench`, plus a tkinter window (`n_in_a_row.gui`).
- A NumPy reinforcement-learning environment (`n_in_a_row.env`) and two
  example policies.
- `n-in-a-row --version`, from `n_in_a_row/version.py`.
- Characterization, engine, agent and CLI tests; Google-style lint (ruff)
  and one CI script, `tests/ci.sh`, run by GitHub Actions and by hand.
- MIT licence.

### Fixed

- The transposition table, which outlives a search, is cleared when the
  board's shape or run length changes, and stores mate scores relative to
  the stored position, so a reused entry no longer misjudges a mate.

[Unreleased]: https://github.com/JJBalisan/n-in-a-row/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/JJBalisan/n-in-a-row/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.4...v1.2.0
[1.1.4]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/JJBalisan/n-in-a-row/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/JJBalisan/n-in-a-row/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/JJBalisan/n-in-a-row/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/JJBalisan/n-in-a-row/releases/tag/v1.0.0
