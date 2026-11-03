"""Checks on the package itself rather than on what it computes."""

from __future__ import annotations

from importlib.metadata import version as installed_version

import preffect


def test_version_is_declared_once_in_effect():
    """The version in the code and the version in pyproject.toml must agree.

    They are written in two places, so they can drift. This is the cheapest
    possible guard against shipping a package whose metadata says one thing and
    whose ``__version__`` says another.
    """
    assert preffect.__version__ == installed_version("preffect")


def test_version_follows_on_from_the_released_one():
    """This is the successor to the released 0.1, so it is numbered 0.2."""
    assert preffect.__version__.startswith("0.2")


def test_public_names_all_import():
    for name in preffect.__all__:
        assert hasattr(preffect, name), f"{name} is in __all__ but is not importable"
