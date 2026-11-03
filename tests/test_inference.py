"""Applying a fitted model, and measuring imputation honestly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from preffect.config import Likelihood, ModelConfig
from preffect.data.masking import mcar_mask
from preffect.data.tensors import TensorData
from preffect.errors import DataError, PreffectError
from preffect.inference import imputation_error, infer
from preffect.model import SimpleModel

N_GENES, N_SAMPLES = 14, 9


def model(**changes) -> SimpleModel:
    torch.manual_seed(0)
    return SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=5, **changes))


def counts() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 50, (N_SAMPLES, N_GENES)).float()


def data() -> TensorData:
    x = counts()
    return TensorData(inputs=x.clone(), targets=x)


# --------------------------------------------------------------------------
# Inference is reproducible.
#
# Version 0.1 ran inference through the training path, which samples from the
# latent distribution, so the same model over the same data gave different
# cleaned counts each time.
# --------------------------------------------------------------------------


def test_the_same_model_and_data_give_the_same_answer_twice():
    m, d = model(), data()
    first, second = infer(m, d), infer(m, d)
    torch.testing.assert_close(first.expected_counts, second.expected_counts)
    torch.testing.assert_close(first.latent, second.latent)
    assert first.deterministic


def test_sampling_the_latent_is_available_and_is_not_deterministic():
    m, d = model(), data()
    first, second = infer(m, d, sample_latent=True), infer(m, d, sample_latent=True)
    assert not torch.allclose(first.expected_counts, second.expected_counts)
    assert not first.deterministic


def test_the_reported_latent_is_the_one_everything_was_built_from():
    """Otherwise the summary and the counts would describe different draws."""
    m, d = model(), data()
    result = infer(m, d)
    m.eval()
    with torch.no_grad():
        rebuilt = m._finish_counts(d.inputs, None, result.latent, result.latent, result.latent)
    torch.testing.assert_close(rebuilt.composition, result.composition)


def test_inference_leaves_the_model_as_it_found_it():
    m = model()
    m.train()
    infer(m, data())
    assert m.training


def test_dropout_is_off_during_inference():
    m = model()
    m.train()  # even when the model was left in training mode
    first, second = infer(m, data()), infer(m, data())
    torch.testing.assert_close(first.expected_counts, second.expected_counts)


# --------------------------------------------------------------------------
# What comes out.
# --------------------------------------------------------------------------


def test_the_result_carries_everything_worth_keeping():
    result = infer(model(), data())
    assert result.expected_counts.shape == (N_SAMPLES, N_GENES)
    assert result.composition.shape == (N_SAMPLES, N_GENES)
    assert result.dispersion.shape == (N_SAMPLES, N_GENES)
    assert result.latent.shape == (N_SAMPLES, 5)
    assert result.library_size.shape == (N_SAMPLES, 1)


def test_zero_log_odds_only_for_the_zero_inflated_likelihood():
    assert infer(model(likelihood=Likelihood.NB), data()).zero_logits is None
    assert infer(model(likelihood=Likelihood.ZINB), data()).zero_logits is not None


def test_it_packages_as_anndata_with_names():
    result = infer(model(), data())
    samples = [f"patient_{i}" for i in range(N_SAMPLES)]
    genes = [f"GENE{i}" for i in range(N_GENES)]
    adata = result.to_anndata(samples, genes)

    assert adata.shape == (N_SAMPLES, N_GENES)
    assert list(adata.obs_names) == samples
    assert list(adata.var_names) == genes
    assert "composition" in adata.layers
    assert adata.obsm["latent"].shape == (N_SAMPLES, 5)
    assert adata.uns["deterministic"] is True


def test_the_names_are_written_as_object_dtype():
    """Finding 48: a fresh install resolves pandas 3 with anndata 0.11, where a
    list of names becomes a string array the writer refuses, and `preffect
    apply` crashed at its last line. Object dtype is written by every anndata."""
    result = infer(model(), data())
    adata = result.to_anndata(
        [f"patient_{i}" for i in range(N_SAMPLES)], [f"GENE{i}" for i in range(N_GENES)]
    )
    assert adata.obs.index.dtype == object, adata.obs.index.dtype
    assert adata.var.index.dtype == object, adata.var.index.dtype
    # And with the default names too.
    unnamed = result.to_anndata()
    assert unnamed.obs.index.dtype == object
    assert unnamed.var.index.dtype == object


def test_the_wrong_number_of_names_is_refused():
    result = infer(model(), data())
    with pytest.raises(DataError, match="names for"):
        result.to_anndata(sample_names=["only_one"])


# --------------------------------------------------------------------------
# Imputation error needs real truth.
#
# Defect 10: version 0.1 looked for a layer called truth, and when it was
# absent the data loader invented one as the per-gene mean. The error routine
# then measured against that, in a column named truth_mu, so on real data it
# reported distance from the gene mean as imputation accuracy.
# --------------------------------------------------------------------------


def test_truth_is_an_argument_with_no_default():
    """There is nowhere for a fabricated truth to come from."""
    import inspect

    signature = inspect.signature(imputation_error)
    assert signature.parameters["truth"].default is inspect.Parameter.empty


def test_a_perfect_prediction_scores_zero():
    x = counts()
    assert imputation_error(x, x)["overall"] == pytest.approx(0.0)


def test_hidden_and_visible_entries_are_reported_separately():
    """Doing well on entries the model was shown is not imputation."""
    x = counts()
    mask = mcar_mask(N_SAMPLES, N_GENES, 0.3, np.random.default_rng(0))
    hidden = torch.from_numpy(mask.hidden)

    predicted = x.clone()
    predicted[hidden] = 0.0  # wrong exactly where it mattered

    scores = imputation_error(predicted, x, hidden=hidden)
    assert scores["visible"] == pytest.approx(0.0)
    assert scores["hidden"] > 0
    assert scores["n_hidden"] == mask.n_hidden


def test_both_kinds_of_error_are_available():
    x = counts()
    predicted = x + 1.0
    squared = imputation_error(predicted, x, kind="mean_squared")["overall"]
    assert squared == pytest.approx(1.0)

    relative = imputation_error(predicted, x, kind="mean_relative")
    assert 0 < relative["overall"] < 1


def test_relative_error_leaves_out_zero_truth_entries_and_says_how_many():
    """Finding 25: dividing by a true value of zero makes the average meaningless.

    Version 0.1 divided by ``truth + 1e-8``, so one zero-truth entry could
    contribute a hundred million. It never showed because version 0.1 compared
    against smoothed per-gene means rather than counts.
    """
    truth = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    predicted = torch.tensor([[7.0, 7.0, 5.0, 5.0]])

    scores = imputation_error(predicted, truth, kind="mean_relative")
    assert scores["zero_truth_entries_skipped"] == 2
    assert scores["fraction_skipped"] == pytest.approx(0.5)
    # Only the two comparable entries count: |5-4|/4 = 0.25 each.
    assert scores["overall"] == pytest.approx(0.25)


def test_relative_error_is_not_a_number_when_every_truth_is_zero():
    scores = imputation_error(torch.ones(2, 2), torch.zeros(2, 2), kind="mean_relative")
    assert np.isnan(scores["overall"])
    assert scores["fraction_skipped"] == pytest.approx(1.0)


def test_an_unknown_kind_is_refused():
    x = counts()
    with pytest.raises(PreffectError, match="mean_squared"):
        imputation_error(x, x, kind="rmsle")


def test_shape_mismatches_are_refused():
    with pytest.raises(PreffectError, match="have to match"):
        imputation_error(torch.zeros(3, 4), torch.zeros(3, 5))
    with pytest.raises(PreffectError, match="have to match"):
        imputation_error(torch.zeros(3, 4), torch.zeros(3, 4), hidden=torch.zeros(2, 4).bool())


def test_an_empty_selection_gives_not_a_number_rather_than_a_crash():
    x = counts()
    nothing_hidden = torch.zeros_like(x, dtype=torch.bool)
    scores = imputation_error(x, x, hidden=nothing_hidden)
    assert np.isnan(scores["hidden"])
    assert scores["visible"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# End to end: a model that has seen masked data imputes better than one that
# has not been trained at all.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_excluded_count_is_over_the_scored_entries():
    """With a mask, the zeros counted are the zeros among the hidden entries, not the matrix's.

    Finding 53: the count used to be taken over the whole matrix before the
    mask was looked at, so a hidden set with no zeros in a matrix full of them
    reported most of the matrix as excluded.
    """
    truth = torch.tensor([[0.0, 5.0, 0.0, 2.0], [0.0, 3.0, 0.0, 4.0]])
    predicted = truth + 1.0
    none_hidden_zero = torch.tensor([[False, True, False, True], [False, True, False, False]])
    scores = imputation_error(predicted, truth, hidden=none_hidden_zero, kind="mean_relative")
    assert scores["zero_truth_entries_skipped"] == 0
    assert scores["fraction_skipped"] == 0.0
    assert scores["n_hidden"] == 3
    one_hidden_zero = torch.tensor([[True, True, False, False], [False, False, False, True]])
    scores = imputation_error(predicted, truth, hidden=one_hidden_zero, kind="mean_relative")
    assert scores["zero_truth_entries_skipped"] == 1
    assert scores["fraction_skipped"] == pytest.approx(1 / 3)
    # Without a mask the whole matrix is what was scored, as before.
    whole = imputation_error(predicted, truth, kind="mean_relative")
    assert whole["zero_truth_entries_skipped"] == 4


def test_training_improves_imputation_of_hidden_entries():
    from pathlib import Path

    from preffect.config import (
        DataConfig,
        LossWeights,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.data.masking import apply_mask
    from preffect.training import train

    rng = np.random.default_rng(0)
    truth = torch.from_numpy(
        rng.negative_binomial(5, 0.1, size=(60, N_GENES)).astype(np.float32)
    )
    mask = mcar_mask(60, N_GENES, 0.2, np.random.default_rng(1))
    hidden = torch.from_numpy(mask.hidden)
    shown = torch.from_numpy(apply_mask(truth.numpy(), mask))

    masked_data = TensorData(inputs=shown, targets=truth, mask=mask)

    config = PreffectConfig(
        data=DataConfig(train_path=Path("/nowhere")),
        model=ModelConfig(hidden_dim=32, latent_dim=8),
        loss=LossWeights(),
        train=TrainConfig(epochs=60, batch_size=20, learning_rate=1e-2),
        runtime=RuntimeConfig(seed=0, num_threads=2),
    )

    torch.manual_seed(0)
    m = SimpleModel(N_GENES, config.model)
    before = imputation_error(infer(m, masked_data).expected_counts, truth, hidden)

    train(m, masked_data, config)
    after = imputation_error(infer(m, masked_data).expected_counts, truth, hidden)

    assert after["hidden"] < before["hidden"]


# --------------------------------------------------------------------------
# infer puts the data where the model is. Finding 45's third half: train moves
# its own copy of the data to the device, the caller's stays on the CPU, and
# the natural next line, infer(model, data), failed with a PyTorch device error.
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_infer_accepts_data_left_on_the_cpu_after_a_gpu_fit(gpu_device):
    import torch

    net = model().to(gpu_device)
    result = infer(net, data())  # data() is on the CPU
    assert result.expected_counts.device.type == "cuda"
    assert torch.isfinite(result.expected_counts).all()


def test_infer_leaves_cpu_data_where_it_is():
    result = infer(model(), data())
    assert result.expected_counts.device.type == "cpu"



# --------------------------------------------------------------------------
# Scoring a predicted network on its edges, not on the loss.
#
# Issue #13: on a network one per cent dense an unweighted cross-entropy is
# almost entirely "no edge here", so the loss cannot say whether the edges are
# being found. These score the edges on a five-sample network whose answers
# were worked out by hand.
# --------------------------------------------------------------------------


def five_sample_network():
    """Edges 0-1, 1-2 and 3-4: three of the ten pairs, so a density of 0.3."""
    observed = torch.zeros(5, 5)
    for a, b in [(0, 1), (1, 2), (3, 4)]:
        observed[a, b] = observed[b, a] = 1.0
    return observed


def symmetric(values: dict[tuple[int, int], float], n: int = 5) -> torch.Tensor:
    predicted = torch.zeros(n, n)
    for (a, b), p in values.items():
        predicted[a, b] = predicted[b, a] = p
    return predicted


def test_network_recovery_on_a_hand_scored_network():
    """Two edges and one non-edge above one half, the third edge below it.

    Predicted: 0-1 at 0.9 and 1-2 at 0.8 are edges, 0-2 at 0.6 is not, 3-4 at
    0.4 is an edge the model missed. So precision 2 of 3, recall 2 of 3, F1
    2/3. Ranked: edge, edge, non-edge, edge, then six non-edges, so the
    precisions at the three edges are 1, 1 and 3/4, averaging 11/12.
    """
    from preffect.inference import network_recovery

    predicted = symmetric({
        (0, 1): 0.9, (1, 2): 0.8, (0, 2): 0.6, (3, 4): 0.4,
        (0, 3): 0.3, (0, 4): 0.25, (1, 3): 0.2, (1, 4): 0.15, (2, 3): 0.1, (2, 4): 0.05,
    })
    scores = network_recovery(predicted, five_sample_network())
    assert scores.n_pairs == 10 and scores.n_edges == 3 and scores.n_predicted == 3
    assert scores.precision == pytest.approx(2 / 3)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.f1 == pytest.approx(2 / 3)
    assert scores.average_precision == pytest.approx(11 / 12)
    assert scores.density == pytest.approx(0.3)
    assert scores.flat_average_precision == pytest.approx(0.3)


def test_network_recovery_is_perfect_when_every_edge_outranks_every_non_edge():
    from preffect.inference import network_recovery

    predicted = symmetric({(0, 1): 0.9, (1, 2): 0.8, (3, 4): 0.7, (0, 2): 0.4})
    scores = network_recovery(predicted, five_sample_network())
    assert (scores.precision, scores.recall, scores.f1) == (1.0, 1.0, 1.0)
    assert scores.average_precision == pytest.approx(1.0)


@pytest.mark.parametrize("flat", [0.0, 0.5, 1.0])
def test_a_flat_prediction_scores_the_density_on_average_precision(flat):
    """No edge anywhere, every edge, or undecided everywhere: none of them
    ranks the pairs, and all score the density, which is the baseline."""
    from preffect.inference import network_recovery

    scores = network_recovery(torch.full((5, 5), flat), five_sample_network())
    assert scores.average_precision == pytest.approx(0.3)
    assert scores.average_precision == pytest.approx(scores.flat_average_precision)


def test_predicting_no_edge_anywhere_scores_zero_on_the_thresholded_measures():
    from preffect.inference import network_recovery

    scores = network_recovery(torch.zeros(5, 5), five_sample_network())
    assert (scores.precision, scores.recall, scores.f1) == (0.0, 0.0, 0.0)
    assert scores.n_predicted == 0


def test_exactly_one_half_is_not_a_predicted_edge():
    """What the collapsed inner-product decoder produces: sigmoid of zero. It
    reads as no edge anywhere rather than as every pair being an edge."""
    from preffect.inference import network_recovery

    scores = network_recovery(torch.full((5, 5), 0.5), five_sample_network())
    assert scores.n_predicted == 0
    assert scores.recall == 0.0


def test_network_recovery_ignores_the_diagonal_and_reads_one_triangle():
    """The diagonal is a sample's link to itself, which is not a pair, and the
    lower triangle repeats the upper, so neither may move a score."""
    from preffect.inference import network_recovery

    predicted = symmetric({(0, 1): 0.9, (1, 2): 0.8, (3, 4): 0.7, (0, 2): 0.4})
    reference = network_recovery(predicted, five_sample_network())
    with_diagonal = predicted.clone()
    with_diagonal.fill_diagonal_(1.0)
    observed_with_self_links = five_sample_network()
    observed_with_self_links.fill_diagonal_(1.0)
    scores = network_recovery(with_diagonal, observed_with_self_links)
    assert scores == reference


def test_network_recovery_refuses_an_asymmetric_network():
    from preffect.inference import network_recovery

    observed = five_sample_network()
    observed[0, 1] = 0.0  # one direction only
    with pytest.raises(PreffectError, match="not symmetric"):
        network_recovery(torch.zeros(5, 5), observed)


def test_network_recovery_refuses_mismatched_shapes():
    from preffect.inference import network_recovery

    with pytest.raises(PreffectError, match="must be square"):
        network_recovery(torch.zeros(4, 5), five_sample_network())


def test_network_recovery_says_what_it_scored():
    from preffect.inference import network_recovery

    text = str(network_recovery(torch.full((5, 5), 0.5), five_sample_network()))
    assert "3 edges among 10 pairs" in text
    assert "flat prediction" in text


# --------------------------------------------------------------------------
# Inference over a masked split uses the corrected library size.
#
# Finding 56. ``infer`` runs the pass twice for a deterministic result, once
# to get the latent mean and once to decode from it, so the correction has to
# reach both or the reported counts would come from an uncorrected second
# pass.
# --------------------------------------------------------------------------


def masked(fraction: float = 0.2) -> tuple[TensorData, torch.Tensor]:
    from preffect.data.masking import apply_mask

    x = counts()
    mask = mcar_mask(N_SAMPLES, N_GENES, fraction, np.random.default_rng(0))
    shown = torch.from_numpy(apply_mask(x.numpy(), mask))
    return TensorData(inputs=shown, targets=x, mask=mask), x


def test_the_reported_library_is_the_corrected_total():
    data_, _ = masked()
    result = infer(model(), data_)
    expected = data_.inputs.sum(dim=1, keepdim=True) * data_.library_scale
    torch.testing.assert_close(result.library_size, expected)


def test_the_corrected_library_is_near_the_true_total_and_the_raw_one_is_not():
    data_, truth = masked()
    corrected = infer(model(), data_).library_size.squeeze(1)
    raw = infer(model(), TensorData(inputs=data_.inputs, targets=truth)).library_size.squeeze(1)
    true_total = truth.sum(dim=1)
    assert float((raw / true_total).median()) < 0.9
    assert float((corrected / true_total).median()) == pytest.approx(1.0, abs=0.1)


def test_drawing_the_latent_uses_the_same_correction():
    """``sample_latent=True`` takes a different branch; it must correct too."""
    data_, _ = masked()
    torch.manual_seed(0)
    drawn = infer(model(), data_, sample_latent=True)
    expected = data_.inputs.sum(dim=1, keepdim=True) * data_.library_scale
    torch.testing.assert_close(drawn.library_size, expected)


def test_with_nothing_hidden_the_result_is_what_it_always_was():
    """``apply`` on a real cohort has no mask, so no number it reports moves."""
    plain = data()
    assert plain.library_scale is None
    result = infer(model(), plain)
    torch.testing.assert_close(result.library_size, plain.inputs.sum(dim=1, keepdim=True))
