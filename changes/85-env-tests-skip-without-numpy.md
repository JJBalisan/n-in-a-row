### Fixed

- The env tests added in #69 (`test_n_env_alternate`,
  `test_n_env_opponent_observe`, `test_n_env_select`) skip without NumPy
  locally, and fail when `CI` is set, like the other env tests, instead
  of erroring on import (#85).
