### Fixed

- `TextAgent` passes `reset` and `observe` on to its fallback agent, so a
  stateful fallback sees every game and move (#40).
