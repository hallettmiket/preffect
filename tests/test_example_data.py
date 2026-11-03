"""The example cohort's generator, without writing any files.

Finding 65: the genes were drawn inside :func:`preffect.example_data.simulate`,
which is called once per split, so a cohort's three splits were three
unrelated sets of genes that shared a set of names. A model fitted on the
training split could not track the test split, and the worked example
presented that as a fact about PREFFECT.

Two things have to hold at once, and they pull against each other:

- the splits of one cohort must be drawn from **the same genes**, which is
  what the fix does; and
- ``simulate`` called once, the way ``exp/08``, ``exp/11``, ``exp/17`` and
  ``exp/19`` call it, must draw **exactly what it always drew**, because those
  experiments' recorded results depend on it.

The digests below are the second of those. They were taken from the generator
as it stood before the fix, so they fail if anything moves the draws.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from preffect.errors import ConfigError, DataError
from preffect.example_data import (
    BATCH_LIFT,
    GenePopulation,
    batch_labels,
    draw_population,
    simulate,
)

# --------------------------------------------------------------------------
# Finding 65, first half: one call draws what it always drew.
# --------------------------------------------------------------------------

#: One call at the shape ``exp/08``, ``exp/11`` and ``exp/17`` use.
ONE_CALL = "5277c303480c7b6ec748c1f44438082d"
#: Two calls from one generator, the second given the first's groups and a
#: lower lift: the shape ``exp/19`` uses for a patient's two tissues.
TWO_CALLS = "47d8a634d7fb3e30b256f5e13466992b"


def digest(counts: np.ndarray, groups: np.ndarray, names: list[str]) -> str:
    """One hash over everything ``simulate`` returns."""
    hasher = hashlib.md5()
    for array in (np.ascontiguousarray(counts), np.ascontiguousarray(groups)):
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    hasher.update("|".join(names).encode())
    return hasher.hexdigest()


def test_one_call_draws_exactly_what_it_drew_before_finding_65():
    """The experiments' synthetic cohorts, unchanged.

    If this fails, ``exp/08``, ``exp/11`` and ``exp/17`` are now reporting
    results on different data than the ones written in their READMEs.
    """
    rng = np.random.default_rng(0)
    assert digest(*simulate(600, 150, rng, offset=0)) == ONE_CALL


def test_two_tissues_from_one_generator_draw_what_they_drew_before():
    """``exp/19``'s pair of tissues, unchanged, including the second's lift."""
    rng = np.random.default_rng(3)
    first, groups, first_names = simulate(80, 150, rng, offset=0)
    second, _, second_names = simulate(80, 150, rng, offset=80, groups=groups, lift=1.8)
    found = digest(np.vstack([first, second]), groups, first_names + second_names)
    assert found == TWO_CALLS


# --------------------------------------------------------------------------
# Finding 65, second half: the splits of one cohort share their genes.
# --------------------------------------------------------------------------


def gene_means(counts: np.ndarray) -> np.ndarray:
    return counts.mean(axis=0)


def test_splits_drawn_from_one_population_agree_about_which_genes_are_high():
    """What the fix is for: the same genes, so a fitted model can be applied."""
    rng = np.random.default_rng(0)
    population = draw_population(150, rng)
    first, _, _ = simulate(100, 150, rng, offset=0, population=population)
    second, _, _ = simulate(100, 150, rng, offset=100, population=population)
    agreement = np.corrcoef(gene_means(first), gene_means(second))[0, 1]
    assert agreement > 0.9, f"the two splits disagree about the genes: {agreement:.3f}"


def test_splits_drawn_without_a_population_do_not_agree():
    """The defect itself, kept as a test so the fix is measured against it."""
    rng = np.random.default_rng(0)
    first, _, _ = simulate(100, 150, rng, offset=0)
    second, _, _ = simulate(100, 150, rng, offset=100)
    agreement = np.corrcoef(gene_means(first), gene_means(second))[0, 1]
    assert agreement < 0.5, (
        f"two independent draws agreed to {agreement:.3f}, which they should not; "
        f"this test is what says the shared population is doing the work"
    )


def test_a_population_of_the_wrong_width_is_refused_by_name():
    rng = np.random.default_rng(0)
    population = draw_population(150, rng)
    with pytest.raises(DataError, match="describes 150 genes and 200 were asked for"):
        simulate(20, 200, rng, offset=0, population=population)


# --------------------------------------------------------------------------
# The batch effect: a batch column that labelled identical samples until
# finding 65, so no correction could change anything.
# --------------------------------------------------------------------------


def test_the_batch_raises_its_own_genes_and_leaves_the_rest_alone():
    """Exact, on the means rather than the counts: one batch-B sample over one
    batch-A sample of the same group is the depth ratio on every gene, times
    the batch lift on the genes the batch raises."""
    rng = np.random.default_rng(0)
    population = draw_population(150, rng)
    groups = np.zeros(40, dtype=int)  # one group, so only the batch differs
    batches = batch_labels(40)
    counts, _, _ = simulate(
        40, 150, rng, offset=0, groups=groups, population=population, batches=batches
    )
    in_b = np.array(batches) == "B"
    shifted = np.zeros(150, dtype=bool)
    shifted[population.batch_genes] = True
    ratio = counts[in_b].mean(axis=0) / counts[~in_b].mean(axis=0)
    lift = ratio[shifted].mean() / ratio[~shifted].mean()
    assert abs(lift - BATCH_LIFT) < 0.1, f"the batch lift reads as {lift:.3f}"


def test_no_batch_labels_means_no_batch_effect():
    """The default, which is what keeps one call bit-identical."""
    population = GenePopulation(
        base=np.full(10, 100.0),
        dispersion=np.full(10, 50.0),
        batch_genes=np.arange(5),
        batch_lift=10.0,
    )
    with_labels, _, _ = simulate(
        200, 10, np.random.default_rng(0), offset=0,
        population=population, batches=batch_labels(200),
    )
    without, _, _ = simulate(
        200, 10, np.random.default_rng(0), offset=0, population=population
    )
    assert with_labels[:, :5].mean() > 1.5 * without[:, :5].mean()
    assert abs(with_labels[:, 5:].mean() / without[:, 5:].mean() - 1) < 0.1


def test_a_batch_effect_without_a_population_is_refused_by_name():
    with pytest.raises(ConfigError, match="needs a population"):
        simulate(10, 20, np.random.default_rng(0), offset=0, batches=batch_labels(10))


def test_one_batch_label_per_sample_is_required():
    rng = np.random.default_rng(0)
    population = draw_population(20, rng)
    with pytest.raises(DataError, match="got 4 batch labels for 10 samples"):
        simulate(10, 20, rng, offset=0, population=population, batches=batch_labels(4))


def test_a_share_that_rounds_down_to_no_genes_is_refused_by_name():
    with pytest.raises(ConfigError, match="rounds down to no genes at all"):
        draw_population(10, np.random.default_rng(0), batch_share=0.01)


def test_a_share_outside_its_range_is_refused_by_name():
    with pytest.raises(ConfigError, match="above 0 and at most"):
        draw_population(10, np.random.default_rng(0), batch_share=1.5)


def test_the_batch_lift_has_to_multiply_rather_than_erase():
    with pytest.raises(ConfigError, match="above 0"):
        draw_population(10, np.random.default_rng(0), batch_lift=0.0)
