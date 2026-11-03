"""Every public name has a docstring, because the docstrings are the reference.

The API reference on the documentation site is generated from the docstrings
by mkdocstrings. There is no second copy to keep in step, which is the point:
version 0.1 committed its built HTML without the sources, so its documentation
can never be corrected. The cost of that arrangement is that a public name
without a docstring is a blank entry on the site, and nothing complains.

Eight of the 155 exported names were blank when this was first run, including
`preffect.cli.main`, which is the entry point of the installed command.

It checks what `__all__` says is public, plus the public methods of exported
classes. A name that is not in `__all__` is internal and may say nothing.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import preffect


def exported_names():
    """Walk the package and yield every name its modules declare public."""
    found = []
    for info in pkgutil.walk_packages(preffect.__path__, "preffect."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            # An optional extra is not installed. The test for that is
            # test_package.py's; this one is about documentation.
            continue
        found.append((info.name, module, None))
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name, None)
            found.append((f"{info.name}.{name}", obj, module))
            if inspect.isclass(obj):
                for attribute, value in vars(obj).items():
                    if attribute.startswith("_") or not callable(value):
                        continue
                    found.append((f"{info.name}.{name}.{attribute}", value, module))
    return found


def test_the_walk_finds_a_substantial_api():
    """Guards the test below: if the walk broke and returned nothing, every
    assertion over it would pass and say nothing."""
    assert len(exported_names()) > 100


def test_nothing_in_all_is_missing():
    """A name in `__all__` that does not exist is a blank entry on the site
    and an ImportError for anyone who reads the reference and tries it."""
    broken = [name for name, obj, module in exported_names()
              if module is not None and obj is None]
    assert broken == [], f"named in __all__ but not defined: {broken}"


@pytest.mark.parametrize("name,obj", [
    (name, obj) for name, obj, _ in exported_names() if obj is not None
], ids=lambda value: value if isinstance(value, str) else "")
def test_every_public_name_has_a_docstring(name, obj):
    assert inspect.getdoc(obj), (
        f"{name} is public and has no docstring, so the API reference shows a "
        f"blank entry for it"
    )
