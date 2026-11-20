### Fixed

- A search stopped by any error or Ctrl-C, not only its time limit, now
  takes back its root move, so `search`, `evaluate_moves` and the labeller
  leave the caller's board as they found it (#32).
