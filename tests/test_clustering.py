"""Grouping samples, and scoring the grouping against known labels.

This is the measure issues #4, #5 and #16 need, because finding 30 established
that a KL divergence cannot rank two models. So these tests matter more than
feature-parity tests usually would: a wrong recovery score would send the
realistic-scale repeats to a wrong conclusion, which is the same shape of
mistake finding 30 was.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scanpy", reason="needs the cluster extra")
pytest.importorskip("sklearn", reason="needs the cluster extra")

from preffect.clustering import (  # noqa: E402
    MIN_RESOLUTION,
    ClusterResult,
    clustering_recovery,
    resolution_for_target,
)
from preffect.errors import PreffectError  # noqa: E402

#: Every test here needs scanpy, matplotlib or umap. See the marker's note
#: in pyproject.toml: the import cost is about 80 seconds and dwarfs the
#: work, so the edit-run loop uses -m 'not extras'.
pytestmark = pytest.mark.extras


def separated_groups(n_groups: int = 4, per_group: int = 30, spread: float = 1.0):
    """Groups far enough apart that the right answer is not in doubt."""
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(n_groups), per_group)
    centres = rng.normal(0, 12, size=(n_groups, 8))
    values = centres[truth] + rng.normal(0, spread, size=(len(truth), 8))
    return values.astype(np.float32), truth


# --------------------------------------------------------------------------
# It finds groups that are really there.
# --------------------------------------------------------------------------


def test_well_separated_groups_are_recovered_exactly():
    values, truth = separated_groups()
    result = resolution_for_target(values, target_clusters=5, seed=0)

    assert result.n_clusters == 4
    assert clustering_recovery(result, truth).adjusted_rand == pytest.approx(1.0)


def test_noise_with_no_groups_does_not_produce_confident_groups():
    """A grouping of structureless data should not score as a recovery."""
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, size=(120, 8)).astype(np.float32)
    invented = np.repeat(np.arange(4), 30)

    result = resolution_for_target(values, target_clusters=5, seed=0)
    assert clustering_recovery(result, invented).adjusted_rand < 0.2


def test_the_same_data_and_seed_cluster_the_same_way_twice():
    values, _ = separated_groups()
    first = resolution_for_target(values, seed=0)
    second = resolution_for_target(values, seed=0)
    assert np.array_equal(first.labels, second.labels)
    assert first.resolution == second.resolution


# --------------------------------------------------------------------------
# The resolution search.
#
# Version 0.1, when the search bottomed out, did:
#
#     elif resolution <= 0.01:
#         resolution = 0.1
#         break
#
# It set the variable and broke WITHOUT re-running Leiden, so the labels came
# from a resolution at or below 0.01 while 0.1 was what got recorded.
# --------------------------------------------------------------------------


def unstructured(n_samples: int = 150, n_features: int = 10):
    """Data with no real groups, where Leiden's count falls as resolution falls.

    ``separated_groups`` returns four clusters at every resolution from 1.0 to
    0.01, so a test using it runs the search 28 times without the search ever
    changing anything. An audit found three tests passing on mutations because
    of that. Finding 37.
    """
    rng = np.random.default_rng(3)
    return rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)


def test_the_search_actually_changes_the_cluster_count():
    """Otherwise every test of the search is testing nothing."""
    result = resolution_for_target(unstructured(), target_clusters=2, seed=0)
    counts = {count for _, count in result.search}
    assert len(counts) > 1, (
        f"the cluster count never moved across {len(result.search)} resolutions, "
        f"so this fixture does not exercise the search at all"
    )


def test_the_reported_resolution_is_the_one_the_labels_came_from():
    """The whole point. Checked against the search path, not against itself."""
    values, _ = separated_groups()
    result = resolution_for_target(values, target_clusters=1, seed=0)

    assert result.hit_the_floor, "this fixture is meant to exhaust the search"

    # Against the labels themselves. Comparing result.resolution to
    # result.search[-1][0] is comparing two copies of the same local, which an
    # audit showed cannot fail: it stayed green while the labels came from the
    # first Leiden run and the resolution from the last. Finding 37.
    assert result.n_clusters == int(np.unique(result.labels).size), (
        "the reported cluster count does not match the labels reported with it"
    )
    last_resolution, last_count = result.search[-1]
    assert result.resolution == last_resolution
    assert result.n_clusters == last_count


def test_running_out_of_room_is_reported_rather_than_hidden():
    values, _ = separated_groups()
    result = resolution_for_target(values, target_clusters=1, seed=0)

    assert result.hit_the_floor
    assert result.n_clusters > 1, "it did not reach the target, and says so"
    assert result.resolution == pytest.approx(MIN_RESOLUTION)


def test_reaching_the_target_is_not_reported_as_running_out_of_room():
    values, _ = separated_groups()
    result = resolution_for_target(values, target_clusters=5, seed=0)
    assert not result.hit_the_floor


def test_every_resolution_tried_is_a_number_someone_could_write_down():
    """Version 0.1 subtracted 0.1 repeatedly and reached 0.30000000000000004.

    That made its own ``resolution <= 0.2`` branch fire one step later than
    intended, at a value nobody would predict from reading the code.
    """
    values, _ = separated_groups()
    result = resolution_for_target(values, target_clusters=1, seed=0)

    # Assert the STEPS, not that a rounded number is round. The record is
    # written as round(resolution, 4), so checking a value read back from it
    # against round(value, 4) is X == X: an audit removed the grid entirely,
    # reproducing version 0.1's 0.30000000000000016, and this test stayed
    # green. Finding 37.
    tried = [resolution for resolution, _ in result.search]
    for resolution in tried:
        assert resolution >= MIN_RESOLUTION
        # Meaningful only because the search record now keeps the raw value.
        # While it stored round(resolution, 4) this assertion was X == X and
        # stayed green with the grid removed entirely. Finding 37.
        assert resolution == round(resolution, 4), (
            f"{resolution!r} is not on the grid, so the stepping is drifting"
        )

    # Deliberately not strict: a list zipped with its own tail is one shorter.
    for earlier, later in zip(tried, tried[1:]):  # noqa: B905
        step = round(earlier - later, 10)
        assert step in (0.1, 0.01) or later == MIN_RESOLUTION, (
            f"step from {earlier} to {later} is {step}, not a grid step"
        )


def test_the_search_only_ever_goes_down():
    values, _ = separated_groups()
    result = resolution_for_target(values, target_clusters=1, seed=0)
    tried = [resolution for resolution, _ in result.search]
    assert tried == sorted(tried, reverse=True)


def test_the_target_is_a_ceiling_not_a_promise():
    """It only lowers the resolution, so it can return far fewer than asked."""
    values, _ = separated_groups(n_groups=2)
    result = resolution_for_target(values, target_clusters=5, seed=0)
    assert result.n_clusters <= 5
    assert not result.hit_the_floor


# --------------------------------------------------------------------------
# Scoring is corrected for chance, and does not care how clusters are numbered.
# --------------------------------------------------------------------------


def test_relabelling_the_clusters_does_not_change_the_score():
    """Cluster 0 has no meaning; a right answer numbered differently is right."""
    truth = np.repeat(np.arange(3), 10)
    renumbered = np.select(
        [truth == 0, truth == 1, truth == 2], [7, 100, -3]
    )
    assert clustering_recovery(renumbered, truth).adjusted_rand == pytest.approx(1.0)


def test_a_random_grouping_scores_near_zero_not_near_one():
    """An uncorrected Rand index would read about 0.7 here and look like success."""
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(5), 40)
    scrambled = rng.permutation(truth)
    assert abs(clustering_recovery(scrambled, truth).adjusted_rand) < 0.05


def test_putting_everything_in_one_cluster_scores_zero():
    truth = np.repeat(np.arange(4), 25)
    assert clustering_recovery(np.zeros(100, dtype=int), truth).adjusted_rand == (
        pytest.approx(0.0)
    )


def test_string_labels_are_fine():
    """Subtypes are names, not numbers."""
    truth = np.array(["LumA", "LumA", "Basal", "Basal", "Her2", "Her2"])
    found = np.array([0, 0, 1, 1, 2, 2])
    assert clustering_recovery(found, truth).adjusted_rand == pytest.approx(1.0)


def test_the_scores_report_what_they_compared():
    truth = np.repeat(np.arange(4), 25)
    scores = clustering_recovery(np.repeat(np.arange(2), 50), truth)
    assert scores.n_true_groups == 4
    assert scores.n_clusters == 2
    assert scores.n_samples == 100
    assert "4 known groups" in str(scores)


def test_a_cluster_result_can_be_scored_directly():
    values, truth = separated_groups()
    result = resolution_for_target(values, seed=0)
    assert clustering_recovery(result, truth).adjusted_rand == pytest.approx(
        clustering_recovery(result.labels, truth).adjusted_rand
    )


# --------------------------------------------------------------------------
# Refusals, rather than quiet nonsense.
# --------------------------------------------------------------------------


def test_mismatched_lengths_are_refused():
    with pytest.raises(PreffectError, match="have to match"):
        clustering_recovery(np.zeros(5, dtype=int), np.zeros(6, dtype=int))


def test_an_impossible_target_is_refused():
    values, _ = separated_groups()
    with pytest.raises(PreffectError, match="at least 1"):
        resolution_for_target(values, target_clusters=0)


def test_one_sample_cannot_be_clustered():
    with pytest.raises(PreffectError, match="at least 2 samples"):
        resolution_for_target(np.zeros((1, 4), dtype=np.float32))


def test_a_matrix_of_the_wrong_shape_is_refused():
    with pytest.raises(PreffectError, match="samples by features"):
        resolution_for_target(np.zeros(10, dtype=np.float32))


def test_labels_have_to_be_one_per_sample():
    with pytest.raises(PreffectError, match="one per sample"):
        ClusterResult(labels=np.zeros((3, 3)), resolution=1.0, n_clusters=1, source="x")


def test_more_neighbours_than_samples_is_handled_not_crashed():
    """A small cohort should not fail because the default asks for ten."""
    values, _ = separated_groups(n_groups=2, per_group=3)
    result = resolution_for_target(values, n_neighbors=10, seed=0)
    assert len(result) == 6


# --------------------------------------------------------------------------
# Clustering a fitted model's output.
# --------------------------------------------------------------------------


def test_each_source_can_be_clustered_and_says_which_it_was():
    import torch

    from preffect.clustering import cluster_samples
    from preffect.config import ModelConfig
    from preffect.data.tensors import TensorData
    from preffect.inference import infer
    from preffect.model import SimpleModel

    torch.manual_seed(0)
    counts = torch.randint(0, 50, (30, 12)).float()
    model = SimpleModel(12, ModelConfig(hidden_dim=16, latent_dim=5))
    result = infer(model, TensorData(inputs=counts.clone(), targets=counts))

    for source in ("latent", "cleaned_counts"):
        clustered = cluster_samples(result, source, seed=0)
        assert clustered.source == source
        assert len(clustered) == 30

    observed = cluster_samples(result, "observed_counts", observed=counts, seed=0)
    assert observed.source == "observed_counts"


def test_clustering_the_observed_counts_needs_them_passed_in():
    """They are the model's input, not its output, so they cannot be inferred."""
    import torch

    from preffect.clustering import cluster_samples
    from preffect.config import ModelConfig
    from preffect.data.tensors import TensorData
    from preffect.inference import infer
    from preffect.model import SimpleModel

    torch.manual_seed(0)
    counts = torch.randint(0, 50, (20, 10)).float()
    model = SimpleModel(10, ModelConfig(hidden_dim=16, latent_dim=4))
    result = infer(model, TensorData(inputs=counts.clone(), targets=counts))

    with pytest.raises(PreffectError, match="not its output"):
        cluster_samples(result, "observed_counts")


def test_an_unknown_source_is_refused_by_name():
    import torch

    from preffect.clustering import cluster_samples
    from preffect.config import ModelConfig
    from preffect.data.tensors import TensorData
    from preffect.inference import infer
    from preffect.model import SimpleModel

    torch.manual_seed(0)
    counts = torch.randint(0, 50, (20, 10)).float()
    model = SimpleModel(10, ModelConfig(hidden_dim=16, latent_dim=4))
    result = infer(model, TensorData(inputs=counts.clone(), targets=counts))

    with pytest.raises(PreffectError, match="observed_counts"):
        cluster_samples(result, "umap")
