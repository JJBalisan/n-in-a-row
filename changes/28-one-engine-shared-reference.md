### Changed

- `labels.ExactLabeller` reuses one engine across positions instead of a
  fresh one each, labelling all 4,520 tic-tac-toe positions about 2.7
  times faster with identical labels; the workaround was no longer needed
  after the v1.0.2 engine fix (#17).
- The engine and harness tests share one brute-force minimax reference,
  `tests/brute_force.py`, and enumerate positions with `labels.reachable`
  (#20).
