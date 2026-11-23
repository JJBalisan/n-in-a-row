### Fixed

- The env tests `test_n_env_alternate`, `test_n_env_opponent_observe` and
  `test_n_env_select` skip without NumPy (and fail when `CI` is set)
  instead of erroring at import (#83).
