### Fixed

- A search cut short by a time limit keeps its last completed depth, so
  the engine no longer plays a losing move and calls it exact (#30).
- The k >= 8 heuristic stays out of the mate band, so it no longer
  reports false forced wins (#30).
- The GUI drops engine moves computed for a game that has been replaced,
  including across the "Cannot load opponent" dialog (#30).
- New game checks the size controls first: bad input shows a dialog and
  leaves the current game and its search alone (#30).
- GUI engine workers get their board and agent up front, and games that
  share an agent no longer search on it at once (#30).
- Under `CI=1` the GUI tests fail, rather than skip, without tkinter or a
  display; hosted CI runs them under xvfb with uv's own Python (#30).
