"""Check for #54: n_in_a_row.__version__ must match pyproject's version.

Stdlib only, and no tomllib (requires-python is >=3.9).
"""

import os
import pathlib
import re
import sys
import unittest

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS))

import n_in_a_row

ROOT = pathlib.Path(_TESTS).parent
_PROJECT_VERSION = re.compile(
    r'^\[project\]$.*?^version\s*=\s*"([^"]+)"', re.MULTILINE | re.DOTALL
)


def _pyproject_version():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = _PROJECT_VERSION.search(text)
    if match is None:
        raise AssertionError("no [project] version in pyproject.toml")
    return match.group(1)


class DunderVersionTest(unittest.TestCase):
    def test_imported_package_is_this_checkout(self):
        here = pathlib.Path(n_in_a_row.__file__).resolve().parent
        self.assertEqual(here, (ROOT / "n_in_a_row").resolve())

    def test_dunder_version_matches_pyproject(self):
        self.assertEqual(n_in_a_row.__version__, _pyproject_version())

    def test_dunder_version_is_a_plain_release_string(self):
        self.assertRegex(n_in_a_row.__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
