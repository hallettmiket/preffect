"""Finding and reading count files."""

from __future__ import annotations

import pytest
from tests.conftest import make_counts

from preffect.data import find_tissue_files, load_tissues
from preffect.errors import DataError


def test_finds_one_file_for_one_tissue(split):
    folder = split("train", [make_counts()])
    files = find_tissue_files(folder, n_tissues=1)
    assert len(files) == 1
    assert files[0].name.endswith("tau_1.h5ad")


def test_returns_tissues_in_order(split):
    folder = split("train", [make_counts(seed=1), make_counts(seed=2), make_counts(seed=3)])
    files = find_tissue_files(folder, n_tissues=3)
    assert [f.name for f in files] == [
        "counts.tau_1.h5ad",
        "counts.tau_2.h5ad",
        "counts.tau_3.h5ad",
    ]


def test_loads_what_it_finds(split):
    folder = split("train", [make_counts(n_samples=5, n_genes=3)])
    tissues = load_tissues(folder, n_tissues=1)
    assert len(tissues) == 1
    assert tissues[0].shape == (5, 3)


# --------------------------------------------------------------------------
# The number of tissues found must match the number asked for.
#
# Defect: version 0.1 silently changed the requested number to whatever it
# found, and the message it logged about doing so printed the already-corrected
# value, so it reported the wrong thing in both directions.
# --------------------------------------------------------------------------


def test_missing_tissue_is_an_error_not_a_silent_adjustment(split):
    folder = split("train", [make_counts()])
    with pytest.raises(DataError, match="no file for tissue number"):
        find_tissue_files(folder, n_tissues=2)


def test_extra_tissue_is_an_error_not_a_silent_truncation(split):
    folder = split("train", [make_counts(seed=1), make_counts(seed=2)])
    with pytest.raises(DataError, match="outside 1 to 1"):
        find_tissue_files(folder, n_tissues=1)


def test_the_error_lists_what_was_actually_there(split):
    folder = split("train", [make_counts()])
    with pytest.raises(DataError) as excinfo:
        find_tissue_files(folder, n_tissues=3)
    assert "counts.tau_1.h5ad" in str(excinfo.value)


def test_two_files_for_the_same_tissue_are_refused(split, tmp_path):
    folder = split("train", [make_counts()])
    make_counts(seed=9).write_h5ad(folder / "other_name.tau_1.h5ad")
    with pytest.raises(DataError, match="more than one file for the same tissue"):
        find_tissue_files(folder, n_tissues=1)


# --------------------------------------------------------------------------
# Subfolders are not searched.
#
# Several real dataset folders hold an `old/` subfolder with previous versions
# of the same files. A recursive search would pick those up instead of, or as
# well as, the current ones.
# --------------------------------------------------------------------------


def test_subfolders_are_not_searched(split):
    folder = split("train", [make_counts(n_samples=6)])
    old = folder / "old"
    old.mkdir()
    make_counts(n_samples=99, seed=7).write_h5ad(old / "counts.tau_1.h5ad")

    files = find_tissue_files(folder, n_tissues=1)
    assert len(files) == 1
    assert files[0].parent == folder
    assert load_tissues(folder, n_tissues=1)[0].n_obs == 6


def test_a_subfolder_alone_is_not_enough(split, tmp_path):
    folder = tmp_path / "train"
    (folder / "old").mkdir(parents=True)
    make_counts().write_h5ad(folder / "old" / "counts.tau_1.h5ad")
    with pytest.raises(DataError, match="subfolders are not searched"):
        find_tissue_files(folder, n_tissues=1)


# --------------------------------------------------------------------------
# Messages name the folder and say what is wrong with it.
# --------------------------------------------------------------------------


def test_missing_folder_says_so(tmp_path):
    with pytest.raises(DataError, match="no such folder"):
        find_tissue_files(tmp_path / "absent", n_tissues=1)


def test_empty_folder_says_so(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(DataError, match="The folder is empty"):
        find_tissue_files(tmp_path / "train", n_tissues=1)


def test_wrongly_named_files_are_listed(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (folder / "counts.h5ad").write_bytes(b"not really h5ad")
    (folder / "notes.txt").write_text("hello")
    with pytest.raises(DataError) as excinfo:
        find_tissue_files(folder, n_tissues=1)
    message = str(excinfo.value)
    assert "counts.h5ad" in message
    assert "tau_<n>.h5ad" in message


def test_unreadable_file_names_itself(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (folder / "broken.tau_1.h5ad").write_bytes(b"this is not an h5ad file")
    with pytest.raises(DataError, match="could not read.*broken.tau_1.h5ad"):
        load_tissues(folder, n_tissues=1)
