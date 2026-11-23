### Added

- `agents.HumanQuit`, also importable as `arena.HumanQuit`: the
  `SystemExit` subclass a `HumanAgent` raises when its player quits or
  input ends. Only it is turned into exit 0; any other agent's
  `SystemExit` exits as before (#38).

### Changed

- A human quit raises `agents.HumanQuit`, a `SystemExit` subclass, in
  place of a plain `SystemExit` (#38).

### Fixed

- `play --opponent human` and `match` with a human seat no longer let
  an undo or quit escape `main` as a traceback; the opponent's undo
  takes back like the player's (#38).
- One agent object in both seats gets `reset` and `observe` once per
  game, not twice (#44).
- `GameRecord.from_json` reads back what `to_json` wrote (#49).
- `bench` prints the k actually played, not `None` (#53).
- `selfplay` builds the two agents of a built-in spec once and reseeds
  them per game, keeping records identical per seed; `module:attribute`
  agents are still built per game (#57).
