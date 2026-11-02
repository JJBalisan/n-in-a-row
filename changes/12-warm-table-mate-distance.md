### Fixed

- A `Negamax` reused across positions reports the fastest mate again: a
  mate stored by a deeper search, but beyond the current iteration's
  horizon, could end iterative deepening on a slower mate (20 of 4,520
  tic-tac-toe positions were a few plies off). Outcomes were always
  right (#12).
