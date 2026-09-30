"""The no-third-party-dependency guarantee, as a test rather than a promise.

The project's central claim is that the runtime imports nothing outside the
Python standard library. That claim is only worth anything if it is checked, so
it is checked here on every supported interpreter -- and the CI workflow runs
this same code.

Why not ``sys.stdlib_module_names``: it was added in Python 3.10, so using it to
assert that a 3.9 job works is self-defeating. That mistake shipped once, was
caught by CI on the 3.9 leg, and is the reason the helper below exists.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import sysconfig
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "llmguard"

# Modules that are part of the standard library but may not appear in every
# source of truth (they are built into the interpreter, or platform-specific).
_BUILTIN_MODULES = {
    "_abc", "_ast", "_bisect", "_blake2", "_bz2", "_codecs", "_collections",
    "_collections_abc", "_compat_pickle", "_compression", "_contextvars",
    "_csv", "_ctypes", "_curses", "_datetime", "_decimal", "_elementtree",
    "_functools", "_hashlib", "_heapq", "_imp", "_io", "_json", "_locale",
    "_lsprof", "_lzma", "_markupbase", "_md5", "_multibytecodec",
    "_opcode", "_operator", "_osx_support", "_pickle", "_posixsubprocess",
    "_py_abc", "_pydecimal", "_pyio", "_queue", "_random", "_sha1", "_sha256",
    "_sha3", "_sha512", "_signal", "_sitebuiltins", "_socket", "_sqlite3",
    "_sre", "_ssl", "_stat", "_statistics", "_string", "_strptime", "_struct",
    "_symtable", "_thread", "_threading_local", "_tokenize", "_tracemalloc",
    "_typing", "_uuid", "_warnings", "_weakref", "_weakrefset", "_winapi",
    "array", "atexit", "builtins", "errno", "faulthandler", "gc", "grp",
    "itertools", "marshal", "math", "nt", "posix", "pwd", "pyexpat",
    "sys", "time", "unicodedata", "winreg", "zipimport", "zlib",
}


def _stdlib_names() -> set:
    """Best available set of standard-library module names on this interpreter.

    ``sys.stdlib_module_names`` (3.10+) is authoritative when present. On 3.9 it
    does not exist, so fall back to the interpreter's own filesystem layout,
    which is exactly the definition of "comes with Python".
    """
    names = set(getattr(sys, "stdlib_module_names", ()) or ())
    names |= set(sys.builtin_module_names)
    names |= _BUILTIN_MODULES
    paths = {sysconfig.get_paths().get("stdlib", ""), sysconfig.get_paths().get("platstdlib", "")}
    for path in filter(None, paths):
        try:
            for entry in pathlib.Path(path).iterdir():
                if entry.name.endswith(".py"):
                    names.add(entry.stem)
                elif entry.is_dir() and (entry / "__init__.py").exists():
                    names.add(entry.name)
        except OSError:
            continue
    names.discard("")
    return names


def find_non_stdlib_imports(package: pathlib.Path) -> list:
    """Return 'file: import x' strings for every non-stdlib import found."""
    allowed = _stdlib_names()
    offenders = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in allowed:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, which is always in-package.
                if node.level == 0 and node.module:
                    root = node.module.split(".")[0]
                    if root not in allowed:
                        offenders.append(f"{path.name}: from {node.module}")
    return offenders


class TestNoThirdPartyDependencies(unittest.TestCase):
    def test_stdlib_detection_helper_works_on_this_interpreter(self):
        """The helper must be usable on 3.9, where stdlib_module_names is absent."""
        names = _stdlib_names()
        for expected in ("json", "sqlite3", "asyncio", "ssl", "unittest", "decimal"):
            self.assertIn(expected, names, f"{expected} missing from stdlib detection")

    def test_package_imports_only_the_standard_library(self):
        offenders = find_non_stdlib_imports(PACKAGE)
        self.assertEqual(
            offenders,
            [],
            "Non-standard-library imports found (this breaks the core guarantee):\n  "
            + "\n  ".join(offenders),
        )

    def test_the_checker_actually_detects_an_offender(self):
        """Guard against the assertion silently passing because it checks nothing."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp) / "pkg"
            fake.mkdir()
            (fake / "__init__.py").write_text("")
            (fake / "bad.py").write_text("import requests\n")
            offenders = find_non_stdlib_imports(fake)
            self.assertTrue(offenders, "checker failed to notice a third-party import")
            self.assertIn("requests", offenders[0])

    def test_declared_dependencies_list_is_empty(self):
        """A populated [project].dependencies would contradict the README."""
        pyproject = PACKAGE.parent / "pyproject.toml"
        self.assertTrue(pyproject.exists(), "pyproject.toml missing")
        text = pyproject.read_text(encoding="utf-8")
        # Match the runtime dependency list specifically, not optional extras.
        self.assertIn("dependencies = []", text, "runtime dependencies must stay empty")

    def test_python_requires_matches_what_ci_tests(self):
        pyproject = PACKAGE.parent / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.9"', text)


if __name__ == "__main__":
    unittest.main()
