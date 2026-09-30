"""Packaging invariants.

Three of these have been wrong at least once already: the distribution name
collided with an unrelated PyPI package, the version string drifted from the
metadata, and the CI matrix caught a stdlib attribute that did not exist on the
oldest supported interpreter. They are cheap to assert and expensive to notice
late.

Stdlib only, and that includes the TOML reading: ``tomllib`` is 3.11+, and this
matrix starts at 3.9. Importing it unconditionally broke the 3.9 and 3.10 legs of
CI, which is exactly the class of mistake these tests exist to prevent, so the
fallback below reads only the handful of keys asserted here rather than pulling in
a dependency.
"""

from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent

try:                                          # Python 3.11+
    import tomllib as _toml

    def _load_toml(text: str) -> dict:
        return _toml.loads(text)

except ModuleNotFoundError:                   # 3.9 and 3.10

    def _load_toml(text: str) -> dict:
        return _parse_toml_subset(text)


def _parse_toml_subset(text: str) -> dict:
    """Read the few keys this file asserts, from ``[project]``.

    Not a TOML implementation. It understands tables, bare and quoted strings,
    booleans, and arrays of strings spanning one or more lines, which is
    everything pyproject.toml uses for the fields checked below. Anything else is
    ignored rather than guessed at.
    """
    out: dict = {}
    table: list = []

    def target() -> dict:
        node = out
        for part in table:
            node = node.setdefault(part, {})
        return node

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\[([^\]]+)\]$", line)
        if m:
            # Nested tables are written [project.scripts], not ["project/scripts"],
            # so the name has to be split into a path.
            table = [part.strip().strip('"') for part in m.group(1).split(".")]
            target()
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*=\s*(.+)$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()

        # A multi-line array: keep reading until the brackets balance.
        while value.count("[") > value.count("]"):
            if i >= len(lines):
                break
            value += " " + lines[i].strip()
            i += 1

        if value.startswith("["):
            target()[key] = re.findall(r'"([^"]*)"', value)
        elif value.startswith('"'):
            target()[key] = value.strip('"')
        elif value in ("true", "false"):
            target()[key] = value == "true"
        else:
            target()[key] = value
    return out


def _metadata() -> dict:
    return _load_toml((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class TestTomlFallback(unittest.TestCase):
    """The fallback must agree with the real parser, or the 3.9 leg is lying."""

    def test_subset_parser_matches_the_real_one(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            self.skipTest("tomllib not available to compare against")
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        real = tomllib.loads(text)
        subset = _parse_toml_subset(text)
        self.assertEqual(subset["project"]["name"], real["project"]["name"])
        self.assertEqual(subset["project"]["version"], real["project"]["version"])
        self.assertEqual(
            subset["project"]["dependencies"], real["project"]["dependencies"]
        )
        self.assertEqual(
            subset["project"]["requires-python"], real["project"]["requires-python"]
        )
        self.assertEqual(subset["project"]["scripts"], real["project"]["scripts"])


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


class TestNothingNeedsModernStdlib(unittest.TestCase):
    """Guards the class of bug that just broke the 3.9 leg.

    An unconditional ``import tomllib`` takes the interpreter down before any test
    can report why, and it only shows up on the two oldest matrix entries. So scan
    the tree for stdlib modules newer than the declared floor and require each use
    to be guarded or explicitly allowed.
    """

    #: module -> the version it first appeared in
    NEWER_STDLIB = {
        "tomllib": (3, 11),
        "graphlib": (3, 9),
        "zoneinfo": (3, 9),
        "importlib.metadata": (3, 8),
    }
    #: Used unguarded on purpose, checked by hand.
    ALLOWED: set = set()

    def test_guarded_or_allowed(self):
        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            if any(part in {".git", "__pycache__", "build", "dist"}
                   for part in path.parts):
                continue
            if path.name == pathlib.Path(__file__).name:
                continue        # this file imports one deliberately, guarded
            source = path.read_text(encoding="utf-8")
            for module in self.NEWER_STDLIB:
                if module in self.ALLOWED:
                    continue
                pattern = rf"^\s*(?:import|from)\s+{re.escape(module)}\b"
                for m in re.finditer(pattern, source, re.MULTILINE):
                    line_start = source.rfind("\n", 0, m.start()) + 1
                    window = source[max(0, line_start - 400):line_start]
                    if "try:" in window or "sys.version_info" in window:
                        continue
                    offenders.append(f"{path.relative_to(ROOT)}: {module}")
        self.assertEqual(sorted(set(offenders)), [], "unguarded newer-stdlib import")


if __name__ == "__main__":
    unittest.main()
