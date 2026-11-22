"""Selection tests for #54: __version__ is right however the package loads.

Written from the issue and the verifier's notes before reading any fix.
Stdlib only, no tomllib (requires-python is >=3.9).
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
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


def _version_in_child(path, prelude=""):
    # -I ignores PYTHON* env and the script dir; -S skips site-packages,
    # so no installed or editable n-in-a-row metadata is visible.
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(path)!r})\n"
        f"{prelude}"
        "import n_in_a_row\n"
        "print(n_in_a_row.__file__)\n"
        "print(n_in_a_row.__version__)\n"
    )
    return subprocess.run(  # noqa: S603 (our own interpreter)
        [sys.executable, "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class UninstalledCheckoutTest(unittest.TestCase):
    def test_checkout_without_metadata_gives_pyproject_version(self):
        proc = _version_in_child(ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        file_line, version_line = proc.stdout.splitlines()[-2:]
        self.assertEqual(
            pathlib.Path(file_line).resolve().parent,
            (ROOT / "n_in_a_row").resolve(),
        )
        self.assertEqual(version_line, _pyproject_version())

    def test_package_copied_outside_repo_imports_with_release_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(
                ROOT / "n_in_a_row",
                pathlib.Path(tmp) / "n_in_a_row",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            proc = _version_in_child(tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertRegex(proc.stdout.splitlines()[-1], r"^\d+\.\d+\.\d+$")

    def test_vendored_copy_ignores_host_project_pyproject(self):
        # Differential: a copy vendored into another project must not
        # report that project's version.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(
                ROOT / "n_in_a_row",
                pathlib.Path(tmp) / "n_in_a_row",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            (pathlib.Path(tmp) / "pyproject.toml").write_text(
                '[project]\nname = "host"\nversion = "9.9.9"\n',
                encoding="utf-8",
            )
            proc = _version_in_child(tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotEqual(proc.stdout.splitlines()[-1], "9.9.9")


class StaleMetadataTest(unittest.TestCase):
    def test_stale_installed_metadata_does_not_leak_into_version(self):
        # A venv synced at an older release reports that release through
        # importlib.metadata while running this checkout's code.
        prelude = (
            "import importlib.metadata as _m\n"
            "_m.version = lambda name: '0.0.1'\n"
        )
        proc = _version_in_child(ROOT, prelude)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.splitlines()[-1], _pyproject_version())


class PublicNameTest(unittest.TestCase):
    def test_version_is_exported_str(self):
        self.assertIn("__version__", n_in_a_row.__all__)
        self.assertIsInstance(n_in_a_row.__version__, str)

    def test_version_is_not_the_stale_values(self):
        self.assertNotIn(n_in_a_row.__version__, ("1.0.0", "1.1.3"))


if __name__ == "__main__":
    unittest.main()
