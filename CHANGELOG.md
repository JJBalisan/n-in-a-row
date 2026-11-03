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

[Unreleased]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.3...HEAD
[1.1.3]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/JJBalisan/n-in-a-row/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/JJBalisan/n-in-a-row/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/JJBalisan/n-in-a-row/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/JJBalisan/n-in-a-row/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/JJBalisan/n-in-a-row/releases/tag/v1.0.0
