"""The simple model's forward pass."""

from __future__ import annotations

import pytest
import torch

from preffect.config import CorrectionVariable, Likelihood, ModelConfig
from preffect.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from preffect.model import SimpleModel

N_GENES, N_SAMPLES = 30, 8
BATCH = CorrectionVariable("batch", "categorical")


def build(**changes) -> SimpleModel:
    config = ModelConfig(hidden_dim=16, latent_dim=6, **changes)
    return SimpleModel(n_genes=N_GENES, config=config)


def counts(n_samples: int = N_SAMPLES, n_genes: int = N_GENES) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 100, (n_samples, n_genes)).float()


def corrections(n: int = 1) -> torch.Tensor:
    return torch.randint(0, 2, (N_SAMPLES, n)).float()


# --------------------------------------------------------------------------
# Shapes and types.
# --------------------------------------------------------------------------


def test_the_pass_produces_every_field_it_promises():
    out = build()(counts())
    assert out.latent_mean.shape == (N_SAMPLES, 6)
    assert out.latent_log_variance.shape == (N_SAMPLES, 6)
    assert out.latent.shape == (N_SAMPLES, 6)
    assert out.composition.shape == (N_SAMPLES, N_GENES)
    assert out.dispersion.shape == (N_SAMPLES, N_GENES)
    assert out.library_size.shape == (N_SAMPLES, 1)
    assert isinstance(out.counts, NegativeBinomial)


def test_the_zero_inflated_likelihood_adds_zero_log_odds():
    out = build(likelihood=Likelihood.ZINB)(counts())
    assert isinstance(out.counts, ZeroInflatedNegativeBinomial)
    assert out.zero_logits is not None
    assert out.zero_logits.shape == (N_SAMPLES, N_GENES)


def test_the_plain_likelihood_has_no_zero_log_odds():
    out = build(likelihood=Likelihood.NB)(counts())
    assert out.zero_logits is None


def test_correction_variables_are_accepted():
    model = build(correction_variables=(BATCH,))
    out = model(counts(), corrections(1))
    assert out.composition.shape == (N_SAMPLES, N_GENES)


def test_the_wrong_number_of_genes_is_refused():
    with pytest.raises(ValueError, match="built for 30 genes"):
        build()(counts(n_genes=12))


# --------------------------------------------------------------------------
# What the pieces mean.
# --------------------------------------------------------------------------


def test_composition_is_a_proportion_across_genes():
    """A softmax over genes, so each sample's predicted proportions sum to one."""
    out = build()(counts())
    totals = out.composition.sum(dim=1)
    torch.testing.assert_close(totals, torch.ones(N_SAMPLES), rtol=1e-5, atol=1e-5)
    assert (out.composition >= 0).all()


def test_dispersion_is_positive():
    """A negative dispersion would make the likelihood meaningless."""
    for log_scale in (True, False):
        out = build(log_scale_dispersion=log_scale)(counts())
        assert (out.dispersion > 0).all()


def test_expected_counts_are_composition_times_library_size():
    out = build()(counts())
    torch.testing.assert_close(
        out.expected_counts, out.composition * out.library_size
    )
    torch.testing.assert_close(out.counts.mu, out.expected_counts)


def test_the_fitted_distribution_can_score_the_counts_it_was_built_from():
    x = counts()
    out = build()(x)
    scores = out.counts.log_prob(x)
    assert scores.shape == x.shape
    assert torch.isfinite(scores).all()


# --------------------------------------------------------------------------
# Library size: observed, or inferred.
# --------------------------------------------------------------------------


def test_by_default_the_library_size_is_the_observed_total():
    x = counts()
    out = build()(x)
    assert not out.infers_library_size
    torch.testing.assert_close(out.library_size, x.sum(dim=1, keepdim=True))


def test_inferring_the_library_size_adds_its_own_latent_space():
    out = build(infer_library_size=True)(counts())
    assert out.infers_library_size
    assert out.library_latent_mean.shape == (N_SAMPLES, 6)
    assert out.library_latent.shape == (N_SAMPLES, 6)
    assert out.predicted_log_library_size.shape == (N_SAMPLES, 1)
    assert (out.library_size > 0).all()


def test_no_library_layers_are_built_when_they_are_not_wanted():
    model = build(infer_library_size=False)
    assert model.library_encoder is None
    assert model.library_decoder is None
    names = {n for n, _ in model.named_parameters()}
    assert not any("library" in n for n in names)


# --------------------------------------------------------------------------
# Ceilings.
# --------------------------------------------------------------------------


def test_the_log_variance_ceiling_is_applied():
    model = build()
    model.config = ModelConfig(hidden_dim=16, latent_dim=6, max_log_variance=0.5)
    out = model(counts())
    assert out.latent_log_variance.abs().max().item() <= 0.5 + 1e-6


def test_extreme_counts_do_not_produce_infinities():
    """A single huge count should not take the whole pass with it."""
    x = counts()
    x[0, 0] = 1e9
    out = build()(x)
    for field in (out.composition, out.dispersion, out.library_size, out.latent):
        assert torch.isfinite(field).all()


# --------------------------------------------------------------------------
# Training is possible: gradients reach everything.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("infer_library", [False, True])
@pytest.mark.parametrize("likelihood", [Likelihood.NB, Likelihood.ZINB])
def test_gradients_reach_every_parameter(infer_library, likelihood):
    model = build(infer_library_size=infer_library, likelihood=likelihood)
    x = counts()
    out = model(x)
    (-out.counts.log_prob(x).sum()).backward()

    without = [n for n, p in model.named_parameters() if p.grad is None]
    assert without == [], f"no gradient reached: {without}"


def test_the_drawn_latent_point_carries_gradient():
    """If reparameterising were done wrongly, the encoder would never learn."""
    model = build()
    out = model(counts())
    out.latent.sum().backward()
    assert model.encoder.to_mean.weight.grad is not None


# --------------------------------------------------------------------------
# The third seam.
# --------------------------------------------------------------------------


def test_no_second_activation_is_applied_to_the_decoded_output():
    """Version 0.1 applied one here on top of the decoder's own.

    exp/03 measured that at the same reconstruction for more than twice the KL
    divergence, so it is gone. Decided in
    decisions/2026-10-15_exp03_second_rectifier_removed.md.
    """
    model = build()
    model.eval()
    hidden = torch.randn(N_SAMPLES, 16) * 3

    assert (hidden < 0).any(), "the fixture produced no negative values to check"
    torch.testing.assert_close(model.activate_decoded(hidden), hidden)


def test_overriding_the_decoded_activation_changes_the_result():
    """The seam still works, so the comparison can be repeated at full scale."""

    class Rectified(SimpleModel):
        def activate_decoded(self, hidden):
            return self.activation(hidden)

    config = ModelConfig(hidden_dim=16, latent_dim=6)
    torch.manual_seed(0)
    plain = SimpleModel(N_GENES, config)
    torch.manual_seed(0)
    varied = Rectified(N_GENES, config)
    plain.eval()
    varied.eval()

    x = counts()
    torch.manual_seed(1)
    a = plain(x).composition
    torch.manual_seed(1)
    b = varied(x).composition
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------
# Finding 30: the KL divergence is not a property of the model alone.
#
# Nothing here is a defect. The invariance is a real property of the
# architecture, and it is tested because a decision was once justified by a KL
# comparison that this property makes close to meaningless.
#
# Shrink the latent by c and multiply the decoder's first layer weight by c:
# the decoder's pre-activation is W'(z/c) + b = Wz + b, unchanged. Everything
# the model predicts is identical. The KL, which reads the latent directly,
# falls by nearly c squared.
# --------------------------------------------------------------------------


def _shrink_latent(model, c: float) -> None:
    """Shrink the latent by ``c``, undoing it in the decoder's first layer."""
    import math

    with torch.no_grad():
        model.encoder.to_mean.weight /= c
        model.encoder.to_mean.bias /= c
        model.encoder.to_log_variance.bias -= 2 * math.log(c)
        model.decoder.layer.weight *= c  # no corrections, so every column


def _gauge_model():
    from preffect.config import ModelConfig
    from preffect.model import SimpleModel

    torch.manual_seed(0)
    return SimpleModel(20, ModelConfig(hidden_dim=16, latent_dim=5))


def test_shrinking_the_latent_leaves_what_the_model_predicts_alone():
    """The core claim, tested on the decoder path with the latent held fixed."""
    model = _gauge_model()
    counts = torch.randint(0, 50, (8, 20)).float()
    torch.manual_seed(7)
    latent = torch.randn(8, 5) * 2.0

    model.eval()
    with torch.no_grad():
        before = model._finish_counts(counts, None, latent, latent, torch.zeros(8, 5))
        _shrink_latent(model, 2.0)
        after = model._finish_counts(
            counts, None, latent / 2.0, latent / 2.0, torch.zeros(8, 5)
        )

    torch.testing.assert_close(after.composition, before.composition)
    torch.testing.assert_close(after.dispersion, before.dispersion)
    torch.testing.assert_close(after.expected_counts, before.expected_counts)


def test_shrinking_the_latent_moves_the_kl_a_long_way():
    """The same transform, and the number used to rank two models does move."""
    from preffect.loss import gaussian_kl_divergence

    model = _gauge_model()
    # Counts at a real cohort's scale (finding 49: up to 2.4 million), so the
    # latent means are substantial. At toy counts of 50 the log-scaled encoder
    # of 2026-10-28 gives means near zero, the variance term dominates the KL,
    # and the shrink moves it the other way; the point here is that it moves.
    counts = torch.randint(0, 50, (8, 20)).float() * 1e4
    model.eval()

    def kl_and_mean_term():
        with torch.no_grad():
            mean, log_variance = model.encoder(counts, None)
            log_variance = log_variance.clamp(-10.0, 10.0)
            return (
                float(gaussian_kl_divergence(mean, log_variance).mean()),
                float(0.5 * mean.pow(2).sum(dim=-1).mean()),
            )

    kl_before, term_before = kl_and_mean_term()
    _shrink_latent(model, 2.0)
    kl_after, term_after = kl_and_mean_term()

    assert kl_after < kl_before * 0.9, "the KL should fall substantially"
    # The mean term falls by exactly 1 / c**2, which is where the gap comes from.
    assert term_after == pytest.approx(term_before / 4.0, rel=1e-4)


def test_the_log_variance_clamp_bounds_how_far_the_latent_can_be_shrunk():
    """The freedom is real but not unlimited, and this is where it stops.

    ``forward`` clamps the log variance to ``max_log_variance``, so once a
    dimension sits on that bound, subtracting from it changes nothing and the
    spread no longer shrinks with the mean. A trained model settles near -6,
    well inside the bound, which is the regime where the invariance holds
    exactly. This test pins the boundary rather than asserting it never bites.
    """
    import math

    model = _gauge_model()
    assert model.config.max_log_variance == 10.0

    at_the_bound = torch.full((4, 5), -10.0)
    shifted = (at_the_bound - 2 * math.log(2.0)).clamp(-10.0, 10.0)
    assert torch.equal(shifted, at_the_bound), "the shift is absorbed by the clamp"

    inside = torch.full((4, 5), -6.0)  # where a trained model sits
    shifted_inside = (inside - 2 * math.log(2.0)).clamp(-10.0, 10.0)
    assert float(shifted_inside[0, 0]) == pytest.approx(-6.0 - 2 * math.log(2.0))


# --------------------------------------------------------------------------
# The library size under a mask.
#
# Finding 56: the library size is the row sum of whatever the model was shown.
# With entries hidden that is short by the hidden fraction, while the targets
# keep the originals, and the composition sums to one across genes, so the
# model is fitted toward a total it structurally cannot reach. The correction
# is a factor supplied with the counts, estimated from how many entries were
# hidden in each sample. Issue #32.
# --------------------------------------------------------------------------


def test_the_library_size_rises_by_the_scale_it_is_given():
    x = counts()
    model = build()
    scale = torch.full((N_SAMPLES, 1), 1.25)
    torch.manual_seed(0)
    plain = model(x).library_size
    torch.manual_seed(0)
    corrected = model(x, library_scale=scale).library_size
    torch.testing.assert_close(corrected, plain * 1.25)


def test_the_expected_counts_rise_with_it_too():
    """The composition sums to one, so the library alone sets the scale."""
    x = counts()
    model = build()
    model.eval()
    torch.manual_seed(0)
    plain = model(x).expected_counts
    torch.manual_seed(0)
    corrected = model(x, library_scale=torch.full((N_SAMPLES, 1), 1.25)).expected_counts
    torch.testing.assert_close(corrected, plain * 1.25)


def test_the_expected_totals_match_the_true_totals_where_they_were_80_per_cent():
    """The issue's own measurement, as a test.

    Synthetic counts with no zeros so nothing else confounds it, a fifth of
    every gene's samples hidden. Before the correction the model's expected
    counts summed to 0.80 of the true total, which is one minus the hidden
    fraction; after it they are within one per cent.

    Four hundred genes rather than this file's thirty, because the entries a
    sample has left are what its total is estimated from, and thirty of them
    is a small sample: the per-sample factor spreads by about a fifth at that
    width and by a twentieth at this one.
    """
    import numpy as np

    from preffect.data.masking import apply_mask, mcar_mask
    from preffect.data.tensors import TensorData
    from preffect.inference import infer

    rng = np.random.default_rng(0)
    n_samples, n_genes = 40, 400
    values = rng.integers(5, 200, size=(n_samples, n_genes)).astype(np.float32)
    mask = mcar_mask(n_samples, n_genes, 0.2, np.random.default_rng(1))
    shown = torch.from_numpy(apply_mask(values, mask))
    truth = torch.from_numpy(values)

    torch.manual_seed(0)
    model = SimpleModel(n_genes=n_genes, config=ModelConfig(hidden_dim=16, latent_dim=6))

    uncorrected = infer(model, TensorData(inputs=shown.clone(), targets=truth))
    corrected = infer(model, TensorData(inputs=shown.clone(), targets=truth, mask=mask))

    true_total = float(truth.sum())
    before = float(uncorrected.expected_counts.sum()) / true_total
    after = float(corrected.expected_counts.sum()) / true_total
    assert before == pytest.approx(0.80, abs=0.01), before
    assert after == pytest.approx(1.00, abs=0.01), after


def test_no_mask_leaves_the_pass_bit_for_bit_what_it_was():
    """``apply`` on real data has no mask, so nothing it produces may move."""
    x = counts()
    model = build()
    model.eval()
    torch.manual_seed(0)
    without = model(x)
    torch.manual_seed(0)
    passing_none = model(x, library_scale=None)
    assert torch.equal(without.library_size, passing_none.library_size)
    assert torch.equal(without.expected_counts, passing_none.expected_counts)
    assert torch.equal(without.composition, passing_none.composition)


def test_an_inferred_library_is_fed_the_corrected_total():
    """The encoder reads the log of the observed total, so it reads the
    corrected one. Otherwise the half of the model that predicts the library
    would be learning from a figure a fifth too small."""
    x = counts()
    model = build(infer_library_size=True)
    model.eval()
    scale = torch.full((N_SAMPLES, 1), 1.25)

    torch.manual_seed(0)
    plain = model(x).library_latent_mean
    torch.manual_seed(0)
    corrected = model(x, library_scale=scale).library_latent_mean
    assert not torch.allclose(plain, corrected), "the correction never reached the encoder"

    # It is the same encoder reading the scaled total, not a different path.
    torch.manual_seed(0)
    directly = model(x * 1.25).library_latent_mean
    torch.testing.assert_close(corrected, directly)
