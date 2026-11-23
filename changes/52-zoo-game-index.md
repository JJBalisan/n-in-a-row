### Fixed

- `Saboteur.throw_games` and `Sandbagger.deviation_games` record the game
  (counted in `reset()`) of each `throws` / `deviations` entry, so entries
  from different games can be told apart (#52).
