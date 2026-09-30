"""Packaging invariants.

Two of these have already been wrong once: the distribution name collided with an
unrelated PyPI package, and the version string drifted from the metadata.
"""

from __future__ import annotations

import pathlib
import re
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


class TestPackaging(unittest.TestCase):
    def test_distribution_name_is_not_the_taken_one(self):
        """`llm-guard` on PyPI is Protect AI's prompt-injection guard."""
        name = _metadata()["project"]["name"]
        self.assertNotEqual(
            name, "llm-guard",
            "this name belongs to an unrelated PyPI project; installing it gets "
            "the wrong software",
        )
        self.assertEqual(name, "llm-cost-guard")

    def test_console_command_is_still_llm_guard(self):
        scripts = _metadata()["project"]["scripts"]
        self.assertEqual(scripts.get("llm-guard"), "llmguard.cli:main")

    def test_version_matches_the_dunder(self):
        meta = _metadata()["project"]["version"]
        init = (ROOT / "llmguard" / "__init__.py").read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
        self.assertIsNotNone(m, "__version__ missing")
        self.assertEqual(m.group(1), meta)

    def test_declares_no_runtime_dependencies(self):
        self.assertEqual(_metadata()["project"]["dependencies"], [])

    def test_requires_python_covers_the_ci_matrix(self):
        spec = _metadata()["project"]["requires-python"]
        self.assertIn("3.9", spec)


if __name__ == "__main__":
    unittest.main()
