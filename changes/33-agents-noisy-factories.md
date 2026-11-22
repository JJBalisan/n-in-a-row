### Fixed

- `NoisyAgent` forwards `reset` and `observe` to the agent it wraps and no
  longer explains a random move with the base agent's rationale (#33).
- `noisy:P:AGENT` no longer edits the agent a `module:attribute` spec
  resolves to, and folds P into the epsilon only of a negamax agent it
  built itself from `perfect`, `negamax:D` or `time:S`, so `noisy:` over a
  `module:attribute` or over another `noisy:` wraps and compounds (#45).
- A `module:factory` spec whose factory returns an `Agent` subclass now
  instantiates it instead of failing on the first move (#46).

### Changed

- `module:factory` specs: a factory or class with an `rng` parameter is
  called with the loader's `rng`, and the zoo factories (`saboteur`,
  `stealth`, `sandbagger`, `locked`, `noisy`) take it, so a seeded run
  plays the same games. Zero-argument factories work as before (#51).
