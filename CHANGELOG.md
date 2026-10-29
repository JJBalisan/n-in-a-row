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

[Unreleased]: https://github.com/JJBalisan/n-in-a-row/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/JJBalisan/n-in-a-row/releases/tag/v1.0.0
