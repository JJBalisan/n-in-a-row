"""Tests for n_in_a_row/version.py and the --version flags that print it.

The three cases every copy of dev-toolchain's templates/version.py gets:
REPO_VERSION pins the output, the real checkout gives a git describe
string, and a copy outside any repository says 'unknown' instead of raising.

Stdlib only:  python3 -m unittest discover -s tests
"""

import contextlib
import importlib.util
import io
import os
import pathlib
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

from n_in_a_row import cli, version

ROOT = pathlib.Path(_TESTS).parent
DESCRIBE = re.compile(r"^(v\d+\.\d+\.\d+-\d+-g)?[0-9a-f]{7,}(-dirty)?$")


def load_copy(path):
    """Imports a copy of the module from its own path, as a copier would."""
    spec = importlib.util.spec_from_file_location("version_copy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRepoVersion(unittest.TestCase):
    def test_env_pins_the_output(self):
        with mock.patch.dict(os.environ, {"REPO_VERSION": "v9.9.9-test"}):
            self.assertEqual(version.repo_version(), "v9.9.9-test")

    def test_empty_env_counts_as_unset(self):
        with mock.patch.dict(os.environ, {"REPO_VERSION": ""}):
            self.assertRegex(version.repo_version(), DESCRIBE)

    def test_real_checkout_is_a_describe_string(self):
        env = {k: v for k, v in os.environ.items() if k != "REPO_VERSION"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertRegex(version.repo_version(), DESCRIBE)

    def test_root_is_the_repo(self):
        self.assertEqual(version.REPO_ROOT, ROOT.resolve())

    def test_outside_a_repository_is_unknown(self):
        env = {k: v for k, v in os.environ.items() if k != "REPO_VERSION"}
        with tempfile.TemporaryDirectory() as tmp:
            copy = pathlib.Path(tmp) / "pkg" / "version.py"
            copy.parent.mkdir()
            shutil.copy(ROOT / "n_in_a_row" / "version.py", copy)
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(load_copy(copy).repo_version(), "unknown")

    def test_missing_git_is_unknown(self):
        env = {k: v for k, v in os.environ.items() if k != "REPO_VERSION"}
        no_pin = mock.patch.dict(os.environ, env, clear=True)
        no_git = mock.patch.object(version, "GIT", "/nonexistent/git")
        with no_pin, no_git:
            self.assertEqual(version.repo_version(), "unknown")


class TestVersionFlag(unittest.TestCase):
    def test_cli_prints_the_pinned_version(self):
        out = io.StringIO()
        pin = mock.patch.dict(os.environ, {"REPO_VERSION": "v1.2.3-0-gabc1234"})
        raises = self.assertRaises(SystemExit)
        with pin, contextlib.redirect_stdout(out), raises as caught:
            cli.main(["--version"])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(out.getvalue(), "n-in-a-row v1.2.3-0-gabc1234\n")


if __name__ == "__main__":
    unittest.main()
