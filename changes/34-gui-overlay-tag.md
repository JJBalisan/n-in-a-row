### Fixed

- The GUI's solver overlay shows a read only on the game, board shape and
  `k` it was computed for, so a late read no longer tags another board
  (#34).
- The GUI keeps the engine's explanation of its move in the detail line
  until the position changes (#50).
- The GUI refuses a size spinbox text over 12 characters before Tcl
  parses it, instead of stalling on a huge number (#63).
