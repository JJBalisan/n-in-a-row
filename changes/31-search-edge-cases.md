### Fixed

- `Negamax.search` reports a won position as a loss for the side to move,
  not a draw (#31).
- `Negamax` rejects a `max_depth` below 1 when it is built, so `negamax:0`
  no longer plays as a search that never looks ahead (#39).
- A search cut by the time limit keeps a win its last pass already
  proved, instead of falling back to an earlier, weaker pass (#62).
