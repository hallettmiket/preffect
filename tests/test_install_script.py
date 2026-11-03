"""The one-step installer, checked in the ways that are cheap to check.

Issue #1. `scripts/install.sh` makes an environment, installs the package and
then checks its own work. These tests do not run it: a real install downloads
torch and takes minutes, which is the wrong thing to put in a suite people run
on every edit. The script was run end to end on a fresh clone by hand before it
was committed, and that is the test that matters.

What is here is the part that goes stale silently. A shell script can be
broken for weeks without anyone noticing, because nobody installs twice, and
the ways it breaks are mostly these: a syntax error from an edit, a rename that
leaves the documentation pointing at a path that no longer exists, and a help
text that drifts from the options the script actually accepts.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install.sh"


def test_the_script_is_there_and_runnable():
    assert SCRIPT.is_file(), "scripts/install.sh is gone and the docs still name it"
    assert SCRIPT.stat().st_mode & 0o111, "scripts/install.sh is not executable"


def test_the_script_parses():
    """`bash -n` reads it without running it, which catches an edit that left
    an unclosed quote or a missing `fi`."""
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_help_runs_and_installs_nothing(tmp_path):
    result = subprocess.run(["bash", str(SCRIPT), "--help"],
                            capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert result.returncode == 0
    assert "--minimal" in result.stdout
    assert list(tmp_path.iterdir()) == [], "--help created something"


def test_an_unknown_option_is_refused_rather_than_ignored(tmp_path):
    """Silently ignoring `--miminal` would install the full set and say
    nothing, which is the wrong direction to fail in."""
    result = subprocess.run(["bash", str(SCRIPT), "--miminal"],
                            capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert result.returncode != 0
    assert "unknown option" in result.stderr


OPTIONS = ["--minimal", "--no-check", "--force", "--python", "--venv", "--help"]


@pytest.mark.parametrize("option", OPTIONS)
def test_every_option_the_help_text_lists_is_accepted(option):
    """The help text and the option parser are two lists of the same thing,
    one in a comment and one in a `case`, and nothing joins them."""
    arms = "\n".join(line for line in SCRIPT.read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    helped = subprocess.run(["bash", str(SCRIPT), "--help"],
                            capture_output=True, text=True, timeout=60).stdout
    assert option in helped, f"{option} is not in the help text"
    # An option that takes a value is parsed as `--python=*)`, so the arm may
    # carry `=*` before its closing bracket.
    assert re.search(rf"{re.escape(option)}(=\*)?\)", arms), (
        f"the help text offers {option} and the option parser does not take it"
    )


def test_the_documentation_points_at_the_script():
    """Two places tell people how to install. Both must name this script, or
    somebody follows the old instructions and never finds out there was an
    easier way."""
    for page in (ROOT / "README.md", ROOT / "docs_src" / "getting-started.md"):
        assert "scripts/install.sh" in page.read_text(), (
            f"{page.name} does not mention the install script"
        )


def test_the_script_does_not_fetch_anything_from_the_internet():
    """It says it will not download and run an installer, and that promise is
    the reason it is safe to tell somebody to run it without reading it."""
    body = SCRIPT.read_text()
    # In the prose it says it will not; make sure it is not also doing it.
    commands = [line for line in body.splitlines()
                if line.strip() and not line.strip().startswith("#")]
    text = "\n".join(commands)
    for fetcher in ("curl ", "wget ", "pip install -U uv", "pip install uv"):
        assert fetcher not in text, f"the script runs {fetcher.strip()}"


def test_a_python_version_that_is_not_installed_is_refused_before_anything_is_made(tmp_path):
    """Finding 64: `--python=VERSION` builds the second library stack, and asking
    for a version that is not there must stop with a sentence rather than fall
    back to whatever Python is on the path, which would silently build the
    wrong stack under the right name."""
    result = subprocess.run(["bash", str(SCRIPT), "--python=9.9", "--no-check"],
                            capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert result.returncode != 0
    assert "no Python 9.9" in result.stderr
    assert not (SCRIPT.parents[1] / ".venv-9.9").exists()
    assert list(tmp_path.iterdir()) == []
