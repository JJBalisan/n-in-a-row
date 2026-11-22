### Fixed

- `solver_targets(depth=N)` labels every position with a fresh search, so a
  depth-limited label no longer depends on which positions were labelled
  before it (#37).
- `PolicyAgent` raises a `ValueError` naming the board when its weights were
  trained for a different board size, instead of NumPy's matmul error (#47).
