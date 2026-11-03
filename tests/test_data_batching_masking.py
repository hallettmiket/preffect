"""Mini-batches, and hiding counts on purpose."""

from __future__ import annotations

import numpy as np
import pytest

from preffect.data import apply_mask, batch_indices, mcar_mask, n_batches
from preffect.errors import ConfigError, DataError

# --------------------------------------------------------------------------
# Every sample is used.
#
# Defect: version 0.1 discarded the final batch of each epoch whenever it was
# smaller than the requested size. With 250 samples and a batch of 200, fifty
# never entered training. On one published dataset it was 13 of 123, every
# epoch, silently.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_samples", "batch_size"),
    [(250, 200), (123, 50), (100, 50), (7, 3), (1, 1), (10, 7)],
)
def test_every_sample_appears_exactly_once(n_samples, batch_size):
    batches = batch_indices(n_samples, batch_size, rng=np.random.default_rng(0))
    seen = np.concatenate(batches)
    assert sorted(seen.tolist()) == list(range(n_samples))


def test_the_short_final_batch_is_kept_by_default():
    batches = batch_indices(250, 200, rng=np.random.default_rng(0))
    assert [len(b) for b in batches] == [200, 50]


def test_dropping_the_short_batch_is_possible_when_asked_for():
    batches = batch_indices(250, 200, rng=np.random.default_rng(0), drop_last=True)
    assert [len(b) for b in batches] == [200]


def test_the_published_case_loses_nobody():
    """123 samples at a batch of 50 lost 23 per epoch in version 0.1."""
    batches = batch_indices(123, 50, rng=np.random.default_rng(0))
    assert sum(len(b) for b in batches) == 123
    assert [len(b) for b in batches] == [50, 50, 23]


def test_batch_count_matches_what_is_produced():
    for n, size, drop in [(250, 200, False), (250, 200, True), (100, 50, False)]:
        expected = n_batches(n, size, drop_last=drop)
        actual = len(batch_indices(n, size, rng=np.random.default_rng(0), drop_last=drop))
        assert expected == actual


def test_a_batch_larger_than_the_data_gives_one_batch():
    batches = batch_indices(10, 500, rng=np.random.default_rng(0))
    assert len(batches) == 1
    assert len(batches[0]) == 10


def test_no_samples_gives_no_batches():
    assert batch_indices(0, 10, rng=np.random.default_rng(0)) == []


# --------------------------------------------------------------------------
# Shuffling is reproducible, and asks for the means to be.
# --------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_batches():
    a = batch_indices(50, 10, rng=np.random.default_rng(3))
    b = batch_indices(50, 10, rng=np.random.default_rng(3))
    assert [x.tolist() for x in a] == [x.tolist() for x in b]


def test_different_seeds_give_different_batches():
    a = batch_indices(50, 10, rng=np.random.default_rng(1))
    b = batch_indices(50, 10, rng=np.random.default_rng(2))
    assert [x.tolist() for x in a] != [x.tolist() for x in b]


def test_shuffling_without_a_generator_is_refused():
    """Version 0.1 shuffled from the global random state, so its runs could not
    be repeated even when a seed had been set elsewhere."""
    with pytest.raises(ConfigError, match="needs a random number generator"):
        batch_indices(50, 10, rng=None, shuffle=True)


def test_unshuffled_batches_need_no_generator():
    batches = batch_indices(10, 4, shuffle=False)
    assert [b.tolist() for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_a_batch_size_below_one_is_refused():
    with pytest.raises(ConfigError, match="at least 1"):
        batch_indices(10, 0, shuffle=False)


# --------------------------------------------------------------------------
# Hiding counts.
# --------------------------------------------------------------------------


def test_each_gene_has_the_same_number_hidden():
    mask = mcar_mask(100, 8, fraction=0.1, rng=np.random.default_rng(0))
    per_gene = mask.hidden.sum(axis=0)
    assert set(per_gene.tolist()) == {10}


def test_the_fraction_hidden_is_what_was_asked_for():
    mask = mcar_mask(200, 10, fraction=0.25, rng=np.random.default_rng(0))
    assert mask.fraction_hidden == pytest.approx(0.25)
    assert mask.n_hidden == 500


def test_hiding_nothing_is_allowed():
    mask = mcar_mask(50, 5, fraction=0.0, rng=np.random.default_rng(0))
    assert mask.n_hidden == 0


def test_a_fraction_too_small_to_hide_anything_says_so():
    """Rounding down to nothing is reported rather than passing as success."""
    with pytest.raises(DataError, match="rounds down to"):
        mcar_mask(10, 5, fraction=0.01, rng=np.random.default_rng(0))


def test_an_impossible_fraction_is_refused():
    with pytest.raises(DataError, match="between 0 and 1"):
        mcar_mask(10, 5, fraction=1.5, rng=np.random.default_rng(0))


def test_the_same_seed_hides_the_same_entries():
    a = mcar_mask(40, 6, 0.25, np.random.default_rng(11))
    b = mcar_mask(40, 6, 0.25, np.random.default_rng(11))
    assert np.array_equal(a.hidden, b.hidden)


def test_applying_a_mask_zeroes_exactly_the_hidden_entries():
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 100, size=(20, 5))  # every entry non-zero
    mask = mcar_mask(20, 5, fraction=0.2, rng=np.random.default_rng(1))

    masked = apply_mask(counts, mask)
    assert np.all(masked[mask.hidden] == 0)
    assert np.array_equal(masked[~mask.hidden], counts[~mask.hidden])


def test_applying_a_mask_leaves_the_original_alone():
    counts = np.ones((10, 4), dtype=np.int64)
    mask = mcar_mask(10, 4, fraction=0.5, rng=np.random.default_rng(0))
    apply_mask(counts, mask)
    assert np.all(counts == 1)


def test_a_mask_of_the_wrong_shape_is_refused():
    counts = np.ones((10, 4))
    mask = mcar_mask(8, 4, fraction=0.5, rng=np.random.default_rng(0))
    with pytest.raises(DataError, match="have to match"):
        apply_mask(counts, mask)


def test_positions_lists_the_hidden_entries():
    mask = mcar_mask(10, 3, fraction=0.2, rng=np.random.default_rng(0))
    positions = mask.positions()
    assert positions.shape == (mask.n_hidden, 2)
    for sample, gene in positions:
        assert mask.hidden[sample, gene]


# --------------------------------------------------------------------------
# The library size under a mask.
#
# Finding 56: the model takes a sample's library size, the total it scales
# every prediction by, from the counts it was shown. With a fifth of the
# entries hidden that total is a fifth short while the targets keep the
# originals, and the composition sums to one, so nothing in the model can make
# the difference up. The mask carries the factor that estimates the real total
# back, from how many entries were hidden and never from their values.
# --------------------------------------------------------------------------


def test_the_scale_is_the_genes_over_the_entries_the_model_was_shown():
    mask = mcar_mask(50, 20, fraction=0.2, rng=np.random.default_rng(0))
    scale = mask.library_scale()
    assert scale.shape == (50, 1)
    expected = 20 / (20 - mask.hidden.sum(axis=1))
    np.testing.assert_allclose(scale.ravel(), expected, rtol=1e-6)


def test_hiding_a_fifth_raises_the_total_by_about_a_quarter():
    """0.8 of the total has to be multiplied by 1.25 to get back to it."""
    mask = mcar_mask(200, 500, fraction=0.2, rng=np.random.default_rng(0))
    assert np.median(mask.library_scale()) == pytest.approx(1.25, abs=0.01)


def test_the_estimated_total_is_within_about_a_per_cent_of_the_true_one():
    """What the correction exists to do, on counts with nothing else in them.

    No zeros, so a hidden entry always removes something and the arithmetic
    is not flattered by entries that were zero anyway. At the 773 genes this
    project selects, the uncorrected total is 0.80 of the true one and the
    corrected one is out by 0.6 per cent in the middle and 3 per cent at
    worst. The estimate gets better with more genes, because the entries
    shown are a larger sample of the row: 1.1 per cent in the middle at 400
    genes, 0.25 per cent at 5000. This test takes the project's own figure.
    """
    n_genes = 773
    rng = np.random.default_rng(0)
    counts = rng.integers(5, 200, size=(60, n_genes)).astype(np.float64)
    mask = mcar_mask(60, n_genes, fraction=0.2, rng=np.random.default_rng(1))
    shown = apply_mask(counts, mask)

    uncorrected = shown.sum(axis=1) / counts.sum(axis=1)
    corrected = (shown.sum(axis=1, keepdims=True) * mask.library_scale()).ravel()
    corrected = corrected / counts.sum(axis=1)

    assert np.median(uncorrected) == pytest.approx(0.80, abs=0.01)
    assert np.median(np.abs(corrected - 1.0)) < 0.01
    assert np.abs(corrected - 1.0).max() < 0.05


def test_the_scale_never_reads_a_hidden_value():
    """Two matrices differing only where they are hidden give the same scale.

    The true total and the visible entries together give the hidden sum
    exactly, so a correction that read the values would be handing the model
    the answer it is scored on.
    """
    rng = np.random.default_rng(0)
    counts = rng.integers(5, 200, size=(30, 40)).astype(np.float64)
    mask = mcar_mask(30, 40, fraction=0.2, rng=np.random.default_rng(1))
    other = counts.copy()
    other[mask.hidden] *= 1000
    np.testing.assert_array_equal(mask.library_scale(), mask.library_scale())
    np.testing.assert_array_equal(
        apply_mask(counts, mask).sum(axis=1), apply_mask(other, mask).sum(axis=1)
    )


def test_mcar_is_the_scheme_a_mask_records():
    assert mcar_mask(10, 4, 0.2, np.random.default_rng(0)).scheme == "mcar"


def test_a_scheme_that_is_not_mcar_is_refused_rather_than_estimated():
    """The estimate is unbiased only when the shown entries are a random
    subsample of the sample's. A scheme that hides entries by their values
    breaks that, and a biased correction is worse than none."""
    from preffect.data.masking import Mask

    mask = Mask(hidden=np.zeros((5, 4), dtype=bool), scheme="mnar")
    with pytest.raises(DataError, match="only valid for the 'mcar' scheme"):
        mask.library_scale()


def test_a_sample_with_nothing_left_visible_is_refused():
    from preffect.data.masking import Mask

    hidden = np.zeros((3, 4), dtype=bool)
    hidden[1] = True
    with pytest.raises(DataError, match="every entry hidden"):
        Mask(hidden=hidden).library_scale()


# --------------------------------------------------------------------------
# The split carries the correction, and so does every batch of it.
#
# A batch has no mask of its own: select drops it, because a mask describes
# the whole split. So the scale is a field rather than something recomputed,
# and finding 56 would come back for mini-batched runs if it were not.
# --------------------------------------------------------------------------


def masked_split(n_samples: int = 20, n_genes: int = 10):
    import torch

    from preffect.data.tensors import TensorData

    rng = np.random.default_rng(0)
    counts = rng.integers(1, 50, size=(n_samples, n_genes)).astype(np.float32)
    mask = mcar_mask(n_samples, n_genes, 0.2, np.random.default_rng(1))
    return TensorData(
        inputs=torch.from_numpy(apply_mask(counts, mask)),
        targets=torch.from_numpy(counts),
        mask=mask,
    ), mask


def test_a_masked_split_fills_in_its_own_correction():
    data, mask = masked_split()
    assert data.library_scale is not None
    np.testing.assert_allclose(
        data.library_scale.numpy(), mask.library_scale(), rtol=1e-6
    )


def test_a_split_with_nothing_hidden_has_no_correction():
    import torch

    from preffect.data.tensors import TensorData

    x = torch.ones(5, 3)
    assert TensorData(inputs=x, targets=x).library_scale is None


def test_a_batch_carries_the_rows_of_the_correction_it_covers():
    data, _ = masked_split()
    batch = data.select(np.array([3, 7, 11]))
    assert batch.mask is None, "a batch has no mask of its own"
    assert batch.library_scale is not None
    np.testing.assert_allclose(
        batch.library_scale.numpy(), data.library_scale.numpy()[[3, 7, 11]], rtol=1e-6
    )


def test_moving_a_split_moves_the_correction_with_it():
    data, _ = masked_split()
    moved = data.to("cpu")
    assert moved.library_scale is not None
    np.testing.assert_allclose(
        moved.library_scale.numpy(), data.library_scale.numpy(), rtol=1e-6
    )


def test_a_correction_of_the_wrong_length_is_refused():
    import torch

    from preffect.data.tensors import TensorData

    x = torch.ones(5, 3)
    with pytest.raises(DataError, match="library_scale has 2 rows"):
        TensorData(inputs=x, targets=x, library_scale=torch.ones(2, 1))
