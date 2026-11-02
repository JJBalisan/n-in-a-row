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
