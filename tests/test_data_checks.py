"""Checks on count data, and the training/validation overlap check."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from tests.conftest import make_counts

from preffect.data import (
    ValidationRepeatsTrainingWarning,
    check_tissues,
    compare_splits,
    problems_with_counts,
    warn_if_validation_repeats_training,
)
from preffect.errors import DataError

# --------------------------------------------------------------------------
# Checks that actually fail.
#
# Defect: version 0.1's check raised an error for each problem, caught that
# same error, wrote it to a log and carried on, then returned True
# unconditionally. Its caller tested the return value, so the check could
# never stop a run. Malformed input was used anyway.
# --------------------------------------------------------------------------


def test_clean_counts_have_no_problems():
    assert problems_with_counts(make_counts(), "training tissue 1") == []


def test_negative_counts_are_a_problem():
    adata = make_counts()
    adata.X = adata.X.astype(np.int64)
    adata.X[0, 0] = -5
    problems = problems_with_counts(adata, "t")
    assert any("cannot be negative" in p for p in problems)


def test_non_integer_counts_are_a_problem():
    adata = make_counts()
    adata.X = adata.X.astype(float)
    adata.X[1, 1] = 2.5
    assert any("whole numbers" in p for p in problems_with_counts(adata, "t"))


def test_missing_values_are_a_problem():
    adata = make_counts()
    adata.X = adata.X.astype(float)
    adata.X[2, 2] = np.nan
    assert any("missing value" in p for p in problems_with_counts(adata, "t"))


@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_duplicate_sample_names_are_a_problem():
    adata = make_counts(sample_names=["a", "b", "c", "a", "e", "f"])
    problems = problems_with_counts(adata, "t")
    # The wording changed in finding 34, from "sample names are not unique" to
    # a message that names the offender and says what to do about it.
    assert any("sample name(s) appear more than once" in p for p in problems)
    assert any("a x2" in p for p in problems)


@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_every_problem_is_reported_not_just_the_first():
    adata = make_counts(sample_names=["a", "a", "c", "d", "e", "f"])
    adata.X = adata.X.astype(float)
    adata.X[0, 0] = -1.5  # negative and not a whole number
    problems = problems_with_counts(adata, "t")
    assert len(problems) >= 3


def test_check_tissues_raises_and_names_the_split():
    adata = make_counts()
    adata.X = adata.X.astype(float)
    adata.X[0, 0] = -1
    with pytest.raises(DataError, match="problem\\(s\\) with the training data"):
        check_tissues([adata], split="training")


def test_check_tissues_passes_clean_data():
    check_tissues([make_counts(seed=1), make_counts(seed=2)], split="training")


def test_tissues_must_share_their_genes():
    a = make_counts(n_genes=4)
    b = make_counts(n_genes=5)
    with pytest.raises(DataError, match="do not share the same genes"):
        check_tissues([a, b], split="training")


# --------------------------------------------------------------------------
# Is the validation set really the training set?
#
# Defect: version 0.1 required a validation folder to exist, so cohorts too
# small to divide used a copy of the training data. Every real dataset it was
# run on had training and validation holding the same samples with the same
# counts, and the reported validation loss was a training loss. Nothing said so.
# --------------------------------------------------------------------------


def test_identical_splits_are_recognised():
    train = make_counts(seed=1)
    overlap = compare_splits(train, train.copy())
    assert overlap.is_effectively_identical
    assert overlap.fraction_repeated == 1.0


def test_identical_splits_warn():
    train = make_counts(seed=1)
    with pytest.warns(ValidationRepeatsTrainingWarning, match="repeats the training data"):
        warn_if_validation_repeats_training([train], [train.copy()])


def test_the_warning_points_at_the_way_out():
    train = make_counts(seed=1)
    with pytest.warns(ValidationRepeatsTrainingWarning) as caught:
        warn_if_validation_repeats_training([train], [train.copy()])
    assert "leave validation_path unset" in str(caught[0].message)


def test_shared_names_with_different_values_are_not_identical():
    """Name overlap alone does not mean leakage.

    Simulated data often numbers its samples '0', '1', '2', so two genuinely
    independent draws share every name while holding different counts. An
    earlier version of this check, run on synthetic data, would have called
    that leakage. It is not.
    """
    names = [str(i) for i in range(6)]
    train = make_counts(seed=1, sample_names=names)
    validation = make_counts(seed=2, sample_names=names)

    overlap = compare_splits(train, validation)
    assert overlap.n_shared_names == 6
    assert not overlap.values_identical_on_shared
    assert not overlap.is_effectively_identical


def test_shared_names_with_different_values_warn_about_nothing():
    """The simple vignette is exactly this case, and it must be quiet.

    Its training and validation files both number their samples 0 to 99, so
    every name is shared, but they are independent draws and no row matches.
    A check keyed on names reported it as fully overlapping. It is not.
    """
    names = [str(i) for i in range(6)]
    train = make_counts(seed=1, sample_names=names)
    validation = make_counts(seed=2, sample_names=names)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        overlaps = warn_if_validation_repeats_training([train], [validation])

    assert overlaps[0].n_shared_names == 6
    assert overlaps[0].n_identical_rows == 0


def test_genuinely_separate_splits_are_quiet():
    train = make_counts(seed=1, sample_names=[f"train_{i}" for i in range(6)])
    validation = make_counts(seed=2, sample_names=[f"val_{i}" for i in range(6)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        overlaps = warn_if_validation_repeats_training([train], [validation])

    assert overlaps[0].n_shared_names == 0
    assert not overlaps[0].is_effectively_identical


def test_partial_overlap_is_reported_with_its_size():
    """Half the training samples genuinely reappear, values and all."""
    import anndata as ad

    train = make_counts(seed=1, n_samples=6, sample_names=[f"s_{i}" for i in range(6)])
    fresh = make_counts(seed=2, n_samples=3, sample_names=[f"s_{i}" for i in range(6, 9)])
    validation = ad.concat([train[["s_0", "s_1", "s_2"]], fresh])

    overlap = compare_splits(train, validation)
    assert overlap.n_identical_rows == 3
    assert not overlap.is_effectively_identical

    with pytest.warns(ValidationRepeatsTrainingWarning, match="3 of 6"):
        warn_if_validation_repeats_training([train], [validation])


def test_every_tissue_is_checked():
    a = make_counts(seed=1)
    b = make_counts(seed=2)
    with pytest.warns(ValidationRepeatsTrainingWarning, match=r"tissues \[1, 2\]"):
        overlaps = warn_if_validation_repeats_training([a, b], [a.copy(), b.copy()])
    assert len(overlaps) == 2
    assert all(o.is_effectively_identical for o in overlaps)


def test_mismatched_tissue_counts_are_refused():
    a = make_counts(seed=1)
    with pytest.raises(ValueError):
        warn_if_validation_repeats_training([a, a.copy()], [a.copy()])


# --------------------------------------------------------------------------
# Repeated names are named, counted, and given a remedy.
#
# Finding 34: the check used to say only "gene names are not unique". True and
# useless. GSE167977, the largest dataset here, has 21 genes all called
# No_Match, the placeholder its preparation wrote where a symbol lookup failed,
# and the message gave no way to find that out.
# --------------------------------------------------------------------------


def test_a_repeated_gene_name_is_named_and_counted():
    adata = make_counts(n_samples=4, n_genes=6)
    adata.var_names = ["A", "No_Match", "C", "No_Match", "No_Match", "F"]

    with pytest.raises(DataError) as raised:
        check_tissues([adata], split="training")

    message = str(raised.value)
    assert "No_Match x3" in message
    assert "1 gene name(s) appear more than once" in message
    assert "covering 3 of 6 columns" in message


def test_the_message_says_what_to_do_about_it():
    adata = make_counts(n_samples=4, n_genes=4)
    adata.var_names = ["A", "A", "C", "D"]

    with pytest.raises(DataError, match="make the names unique"):
        check_tissues([adata], split="training")


def test_several_repeated_names_are_summarised_worst_first():
    adata = make_counts(n_samples=4, n_genes=9)
    adata.var_names = ["x", "x", "x", "x", "y", "y", "y", "z", "z"]

    with pytest.raises(DataError) as raised:
        check_tissues([adata], split="training")

    message = str(raised.value)
    assert message.index("x x4") < message.index("y x3") < message.index("z x2")


def test_many_repeated_names_are_truncated_with_a_count():
    adata = make_counts(n_samples=3, n_genes=10)
    adata.var_names = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]

    with pytest.raises(DataError, match="and 2 more"):
        check_tissues([adata], split="training")


def test_repeated_sample_names_are_reported_the_same_way():
    adata = make_counts(n_samples=4, n_genes=3)
    adata.obs_names = ["p1", "p1", "p3", "p4"]

    with pytest.raises(DataError, match=r"sample name\(s\) appear more than once"):
        check_tissues([adata], split="training")


def test_unique_names_raise_nothing():
    adata = make_counts(n_samples=4, n_genes=4)
    adata.var_names = ["A", "B", "C", "D"]
    check_tissues([adata], split="training")
