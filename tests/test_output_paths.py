"""Where a run is allowed to write, checked before it does the work.

Finding 28: an experiment trained eight seeds, built its report, and threw it
all away on the last statement because `--out` was a directory and the writer
wanted a file. Ten minutes of work lost to a check that could have happened
immediately.

Three guards came out of that, in three places, and none of them had a test
until this file. The house rule is a test for every defect, and finding 28 had
been treated as fixed on the strength of the code reading correctly.

- :func:`preffect.cli.check_writable` probes the destination before training.
- ``exp/common.py``'s ``write_report`` accepts a directory and versions a file
  into it, which is what the append_only layout makes natural to pass.
- ``exp/06``'s ``resolve_output`` additionally refuses anywhere inside this
  repository, because that experiment writes participant-level counts.

The ordering matters as much as the refusal. A guard that runs after the
training is the bug, not the fix, so the last test here runs the script and
checks it refuses before it has even looked for the data.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from preffect.cli import check_writable
from preffect.errors import PreffectError

ROOT = Path(__file__).resolve().parents[1]


def _experiments_or_skip() -> Path:
    """exp/ is part of the development record, not of a release; a copy without it skips."""
    folder = ROOT / "exp"
    if not folder.is_dir():
        pytest.skip("exp/ is part of the development record, not of a release")
    return folder


def load_make_counts():
    """Import exp/06's make_counts.py without running it."""
    path = ROOT / "exp/06_pam50_subtyping_replication/make_counts.py"
    sys.path.insert(0, str(_experiments_or_skip()))
    spec = importlib.util.spec_from_file_location("exp_06_make_counts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# check_writable: the command line's guard
# --------------------------------------------------------------------------


def test_a_directory_where_a_file_was_meant_is_refused(tmp_path):
    """The exact shape of finding 28, now caught before any work happens."""
    with pytest.raises(PreffectError, match="which is a directory"):
        check_writable(tmp_path, "--out")


def test_a_missing_parent_folder_is_created_rather_than_refused(tmp_path):
    """Refusing this would be unhelpful: a new folder is almost always meant."""
    destination = tmp_path / "new" / "deeper" / "model.pth"
    check_writable(destination, "--out")
    assert destination.parent.is_dir()


def test_an_unwritable_folder_is_refused_before_the_work(tmp_path):
    """Permissions are probed rather than assumed, which is the only way to
    know: the folder exists and is readable, and still cannot be written."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(PreffectError, match="cannot write"):
            check_writable(locked / "model.pth", "--out")
    finally:
        locked.chmod(0o700)


def test_the_probe_file_does_not_survive_the_check(tmp_path):
    """A guard that litters is a guard people turn off."""
    check_writable(tmp_path / "model.pth", "--out")
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# resolve_output: exp/06's extra refusal, because it writes participant counts
# --------------------------------------------------------------------------


def test_a_path_inside_the_repository_is_refused(tmp_path):
    """This experiment writes participant-level counts with sample
    identifiers, and the repository is meant to go public."""
    module = load_make_counts()
    with pytest.raises(SystemExit, match="inside the repository"):
        module.resolve_output(ROOT / "data" / "counts.h5ad")


def test_a_path_that_only_looks_like_it_is_inside_is_allowed(tmp_path):
    """The refusal is about where the file lands, not how the path is spelt.

    A path written as the repository followed by enough `..` to climb out of
    it starts with the repository's own name and ends up nowhere near it.
    Comparing the text would refuse it, which is the opposite mistake and just
    as annoying: it stops somebody writing where they are entitled to. This is
    why `resolve_output` resolves before it compares.
    """
    module = load_make_counts()
    walk_out = ROOT / os.path.relpath(tmp_path, ROOT) / "counts.h5ad"
    assert str(walk_out).startswith(str(ROOT)), "the test's own premise"
    assert module.resolve_output(walk_out) == (tmp_path / "counts.h5ad").resolve()


def test_a_path_written_the_long_way_round_is_still_refused(tmp_path):
    """And the same resolution in the direction that matters more."""
    module = load_make_counts()
    with pytest.raises(SystemExit, match="inside the repository"):
        module.resolve_output(ROOT / "exp" / ".." / "data" / "counts.h5ad")


def test_a_path_outside_the_repository_is_honoured(tmp_path):
    """Somebody may have a good reason, so this refuses a place, not a habit."""
    module = load_make_counts()
    destination = module.resolve_output(tmp_path / "elsewhere" / "counts.h5ad")
    assert destination == (tmp_path / "elsewhere" / "counts.h5ad").resolve()
    assert destination.parent.is_dir()


def test_no_out_and_no_data_root_refuses_rather_than_guessing(monkeypatch):
    """Guessing would put participant counts somewhere nobody chose."""
    module = load_make_counts()
    monkeypatch.delenv("MURMURENT_DATA_ROOT", raising=False)
    with pytest.raises(SystemExit, match="nowhere safe to write"):
        module.resolve_output(None)


def test_the_default_lands_under_append_only(tmp_path, monkeypatch):
    """Where the hooks that refuse deletes and overwrites actually apply."""
    module = load_make_counts()
    monkeypatch.setenv("MURMURENT_DATA_ROOT", str(tmp_path))
    destination = module.resolve_output(None)
    assert destination.parent == (
        tmp_path / "append_only" / "preffect_development" / "06_pam50_subtyping_replication"
    )
    assert destination.parent.is_dir()


# --------------------------------------------------------------------------
# write_report: a directory is the natural thing to pass, so it is accepted
# --------------------------------------------------------------------------


def a_small_report() -> dict:
    """The least a report can be and still print its table."""
    return {
        "question": "does anything change",
        "summary": {
            "final_loss": {
                "kept": {"mean": 2.0, "values": [1.0, 2.0, 3.0]},
                "removed": {"mean": 2.1, "values": [1.1, 2.1, 3.1]},
            }
        },
    }


def test_write_report_versions_into_a_directory(tmp_path):
    """Finding 28 itself: passing the append_only folder used to raise
    IsADirectoryError after every seed had trained."""
    sys.path.insert(0, str(_experiments_or_skip()))
    from common import write_report

    first = write_report(a_small_report(), "03_decoder_rectifier_removal", tmp_path)
    second = write_report(a_small_report(), "03_decoder_rectifier_removal", tmp_path)
    assert first.is_file() and second.is_file()
    assert first != second, "the second report overwrote the first"


def test_write_report_never_overwrites_an_existing_report(tmp_path):
    """append_only means append only, and a rerun is the common case."""
    sys.path.insert(0, str(_experiments_or_skip()))
    from common import write_report

    paths = [write_report(a_small_report(), "03_decoder_rectifier_removal", tmp_path)
             for _ in range(3)]
    assert len(set(paths)) == 3
    assert all(p.is_file() for p in paths)


# --------------------------------------------------------------------------
# The ordering, which is the half that matters
# --------------------------------------------------------------------------


def test_exp_06_refuses_the_path_before_it_looks_for_the_data():
    """A guard that runs after the training is finding 28, not its fix.

    Run the script for real with an output path it must refuse and a data
    folder that does not exist. If the path check comes first, the message is
    about the path. If it has drifted below the data loading, the message is
    about the missing data, and a real run would train for an hour before
    refusing.
    """
    _experiments_or_skip()
    environment = dict(os.environ, MURMURENT_DATA_ROOT="")
    result = subprocess.run(
        [sys.executable, str(ROOT / "exp/06_pam50_subtyping_replication/make_counts.py"),
         "--out", str(ROOT / "data" / "counts.h5ad"),
         "--data", "/nonexistent/no/such/dataset",
         "--epochs", "1"],
        capture_output=True, text=True, cwd=ROOT, env=environment, timeout=300,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "inside the repository" in combined, (
        f"expected the output path to be refused first; got:\n{combined[-2000:]}"
    )
    assert not (ROOT / "data" / "counts.h5ad").exists()
