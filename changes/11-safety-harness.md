### Added

- `n_in_a_row.harness`, a shared harness for AI-safety experiments with the
  solver as ground truth: an exact move labeller with an SQLite cache,
  depth-d search monitors, regret metrics, tie sets and covert capacity,
  AUROC / precision / power / CUSUM statistics, saboteur, sandbagger and
  password-locked agents, and text play for language models (#11).
- `n_in_a_row.harness.trainer`: the `learn_from_solver.py` distillation
  pipeline as a library, with pluggable teachers; both examples now use it
  and behave as before (#11).
