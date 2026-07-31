"""Guards on the declared dependencies (issue #92).

The PyQt6 requirement had no upper bound, so every fresh install silently took
whatever Qt minor had shipped most recently -- including ones the CI matrix has never
run. This asserts the ceiling is still there, because it is a one-character deletion
away from being gone and nothing else would notice.
"""

from pathlib import Path

# tomllib is stdlib only from 3.12; the project floor is 3.10, so fall back to the
# `tomli` backport (a dev dependency on those versions), matching
# tests/unit/test_annotation_types.py.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - version-dependent
    import tomli as tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dependencies():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["dependencies"]


def _requirement(name):
    prefix = name.lower()
    for entry in _dependencies():
        # Split on the first comparison character; the name is everything before it.
        head = entry.split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("~")[0]
        if head.strip().lower() == prefix:
            return entry
    return None


def test_pyqt6_is_declared():
    assert _requirement("PyQt6") is not None


def test_pyqt6_has_an_upper_bound():
    """Without a ceiling, an untested Qt minor reaches users the day it is released.

    The bound is a tested ceiling, NOT the fix for issue #92 -- that report is DLL
    shadowing inside a Conda environment (ADR-046). If this test ever fails because
    someone tightened the cap to work around #92, read the comment in pyproject.toml
    before changing it.
    """
    requirement = _requirement("PyQt6")
    assert "<" in requirement, (
        f"PyQt6 requirement {requirement!r} has no upper bound"
    )


def test_pyqt6_floor_still_admits_the_documented_minimum():
    """docs/02_architecture_constraints.md and ADR-014 both promise 6.7 as the floor."""
    assert ">=6.7" in _requirement("PyQt6")


@pytest.mark.parametrize("name", ["ultralytics", "PyQt6"])
def test_the_bounded_dependencies_stay_bounded(name):
    """Both carry a ceiling for the same reason: a major/minor bump upstream has
    broken this app before, and neither is exercised by CI until someone upgrades."""
    assert "<" in _requirement(name)
