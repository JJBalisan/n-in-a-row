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
