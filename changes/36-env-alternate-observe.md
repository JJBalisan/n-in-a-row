### Fixed

- `BatchEnv` swaps each game's side every time that game finishes,
  instead of sharing one counter across games that finish in the same
  step (#36).
- `TicTacToeEnv` calls `opponent.observe` after every move by either
  side, as `arena.play_game` does (part of #42; the unused `seed`
  argument is still open).
