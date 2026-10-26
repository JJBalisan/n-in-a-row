#!/bin/bash
#
# CI for wt-tictactoe, drafted 2026-10-26 by dev-toolchain's ci/gen_ci_sh.py
# from ci.yml.
# Review it, then make the workflow's one step `bash tests/ci.sh`; from then
# on edit this file, not the workflow.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# job: test (ci.yml)
# runs-on: ubuntu-latest

# step: actions/checkout@v7
# uses: actions/checkout@v7 (nothing needed: the repo is already checked out)

# step: astral-sh/setup-uv@1c37ad07a6a961277cf70c0d37d6f313000f5884
# uses: astral-sh/setup-uv@1c37ad07a6a961277cf70c0d37d6f313000f5884 (the local environment must provide: uv on PATH)

# step: Format (ruff format --check)
uvx ruff@0.16.10 format --check .

# step: Lint (ruff check)
uvx ruff@0.16.10 check --output-format github .

# step: Tests (unittest, with the test extra)
uv sync --locked --extra test --python 3.12
uv run --python 3.12 python -m unittest discover -s tests -v
