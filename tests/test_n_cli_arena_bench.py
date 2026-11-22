"""#53: bench's first line names the k actually played, never None."""

import contextlib
import io
import unittest

from n_in_a_row import cli

_TINY = ["--games", "1", "--depth", "1", "--batch", "1", "--steps", "1"]


def _first_line(extra):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.main(["bench", *extra, *_TINY])
    return out.getvalue().splitlines()[0]


class TestBenchHeader(unittest.TestCase):
    def test_default(self):
        self.assertEqual(_first_line([]), "board 3x3, 3 in a row")

    def test_size(self):
        self.assertEqual(_first_line(["--size", "4"]), "board 4x4, 4 in a row")

    def test_rows_cols_uses_shorter_side(self):
        self.assertEqual(
            _first_line(["--rows", "5", "--cols", "3"]),
            "board 5x3, 3 in a row",
        )

    def test_explicit_k_kept(self):
        self.assertEqual(
            _first_line(["--size", "4", "-k", "3"]), "board 4x4, 3 in a row"
        )


if __name__ == "__main__":
    unittest.main()
