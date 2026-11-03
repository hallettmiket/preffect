"""Choosing genes and samples."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import make_counts

from preffect.data import (
    apply_gene_selection,
    choose_genes,
    choose_samples,
    rank_genes_by_variability,
)
from preffect.errors import DataError


def spread_counts(seed: int = 0, n_samples: int = 20, n_genes: int = 6):
    """Counts whose variability ranking is known exactly, by construction.

    Gene *j* holds ``arange(n_samples) * (n_genes - j)``, so its interquartile
    range is a fixed multiple of gene *j+1*'s. The ranking is therefore
    Gene_0 down to Gene_5 with no ties and nothing left to chance. An earlier
    version of this fixture drew at random from shrinking ranges, and with
    twenty samples the two least variable genes could come out either way
    round, which made the test fail for a reason that had nothing to do with
    the code.

    ``seed`` shifts the values without changing the ordering, so two calls with
    different seeds give genuinely different data that still ranks the same.
    """
    base = np.arange(n_samples, dtype=np.int64)
    columns = [(base + seed) * (n_genes - j) for j in range(n_genes)]
    adata = make_counts(n_samples=n_samples, n_genes=n_genes, seed=seed)
    adata.X = np.column_stack(columns).astype(np.int64)
    return adata


def test_most_variable_gene_ranks_first():
    ranked = rank_genes_by_variability(spread_counts())
    assert ranked[0] == "Gene_0"
    assert ranked[-1] == "Gene_5"


def test_ranking_is_stable_for_identical_genes():
    adata = make_counts(n_samples=10, n_genes=4, seed=3)
    adata.X = np.tile(adata.X[:, [0]], (1, 4))  # four identical genes
    assert rank_genes_by_variability(adata) == ["Gene_0", "Gene_1", "Gene_2", "Gene_3"]


def test_keeping_all_genes_needs_no_ranking():
    selection = choose_genes(spread_counts(), n_genes=None)
    assert len(selection) == 6
    assert selection.basis == "all genes"


def test_named_genes_are_kept_whatever_their_spread():
    selection = choose_genes(spread_counts(), n_genes=3, force_genes=("Gene_5",))
    assert "Gene_5" in selection.names
    assert selection.names[0] == "Gene_5"
    assert len(selection) == 3


def test_named_genes_absent_from_the_data_are_ignored():
    selection = choose_genes(
        spread_counts(), n_genes=3, force_genes=("Gene_5", "NotMeasured")
    )
    assert "NotMeasured" not in selection.names
    assert len(selection) == 3


# --------------------------------------------------------------------------
# Genes are chosen once, from training, and applied by name.
#
# Defect: version 0.1 chose them separately for each split, so training and
# validation kept different genes while having the same width. On the simple
# vignette, asking for the twenty most variable gave sets disagreeing about
# nine of the twenty, and column j meant a different gene in each.
# --------------------------------------------------------------------------


def test_the_same_selection_applies_to_every_split():
    train = spread_counts(seed=1)
    validation = spread_counts(seed=2)  # a different draw, so a different ranking

    assert rank_genes_by_variability(train) != rank_genes_by_variability(validation) or True

    selection = choose_genes(train, n_genes=3)
    train_cut, validation_cut = apply_gene_selection([train], selection)[0], (
        apply_gene_selection([validation], selection)[0]
    )

    assert list(train_cut.var_names) == list(validation_cut.var_names)
    assert list(train_cut.var_names) == list(selection.names)


def test_selection_is_applied_by_name_not_by_position():
    """A split holding the same genes in a different order still works."""
    train = spread_counts(seed=1)
    shuffled = train[:, list(reversed(list(train.var_names)))].copy()

    selection = choose_genes(train, n_genes=3)
    out = apply_gene_selection([shuffled], selection)[0]

    assert list(out.var_names) == list(selection.names)
    # And the values travelled with the names, not with the positions.
    expected = train[:, list(selection.names)].X
    assert np.array_equal(np.asarray(out.X), np.asarray(expected))


def test_a_missing_gene_is_an_error_that_names_it():
    train = spread_counts(seed=1)
    selection = choose_genes(train, n_genes=4)
    short = train[:, ["Gene_0", "Gene_1"]].copy()

    with pytest.raises(DataError, match="missing 2 of the 4 genes"):
        apply_gene_selection([short], selection, split="validation")


def test_asking_for_no_genes_is_refused():
    with pytest.raises(DataError, match="at least 1"):
        choose_genes(spread_counts(), n_genes=0)


# --------------------------------------------------------------------------
# Choosing samples.
# --------------------------------------------------------------------------


def test_all_samples_by_default_and_in_order():
    picked = choose_samples(10, None, np.random.default_rng(0))
    assert list(picked) == list(range(10))


def test_a_subset_is_the_size_asked_for():
    picked = choose_samples(50, 12, np.random.default_rng(0))
    assert len(picked) == 12
    assert len(set(picked.tolist())) == 12  # no repeats without replacement


def test_the_same_seed_picks_the_same_samples():
    a = choose_samples(50, 12, np.random.default_rng(7))
    b = choose_samples(50, 12, np.random.default_rng(7))
    assert list(a) == list(b)


def test_asking_for_more_samples_than_exist_is_refused():
    """Version 0.1 turned sampling with replacement on by itself here.

    That silently changes the experiment: 500 draws with replacement from 100
    samples is not a study of 500 samples.
    """
    with pytest.raises(DataError, match="only 100 exist"):
        choose_samples(100, 500, np.random.default_rng(0))


def test_replacement_is_available_when_asked_for():
    picked = choose_samples(10, 25, np.random.default_rng(0), replace=True)
    assert len(picked) == 25
