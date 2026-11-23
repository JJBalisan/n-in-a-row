### Fixed

- `Saboteur.throw_games` and `Sandbagger.deviation_games` record the game
  of each `throws` / `deviations` entry, so entries from different games
  can be told apart. The game is the agent's new `games` attribute, which
  each `reset()` raises by one from 0: entries logged before the first
  `reset()` get 0, and with one `reset()` per game the first game is 1
  (#52).
