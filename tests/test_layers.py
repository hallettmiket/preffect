"""The network pieces: shapes, gradients, and building only what is used."""

from __future__ import annotations

import pytest
import torch

from preffect.layers import (
    CountDecoder,
    CountEncoder,
    FiLM,
    LibrarySizeDecoder,
    LibrarySizeEncoder,
)

N_GENES, N_SAMPLES = 40, 7
HIDDEN, LATENT = 12, 5


def corrections(n_corrections: int, n_samples: int = N_SAMPLES):
    return torch.randint(0, 2, (n_samples, n_corrections)).float()


# --------------------------------------------------------------------------
# Only what is used gets built.
#
# Defect: version 0.1's encoder constructed the graph path and the plain path
# together. For a simple model on 20,000 genes that allocated a graph
# attention layer of 16 million parameters it never touched, measured at 94%
# of the encoder's weights and 64 MB, saved into every session file.
# --------------------------------------------------------------------------


def test_the_simple_encoder_has_no_graph_layers():
    encoder = CountEncoder(
        n_genes=20_000, n_corrections=1, hidden_dim=50, latent_dim=20
    )
    names = {name for name, _ in encoder.named_modules()}
    assert not any("gat" in n.lower() or "graph" in n.lower() for n in names)


def test_the_simple_encoder_is_the_size_its_layers_imply():
    """Every parameter belongs to a layer the forward pass uses."""
    n_genes, n_corr, hidden, latent = 20_000, 1, 50, 20
    encoder = CountEncoder(n_genes, n_corr, hidden, latent)

    wide = hidden + n_corr
    expected = (
        (n_genes + n_corr) * wide + wide          # layer1
        + wide * hidden + hidden                  # layer2
        + hidden * latent + latent                # to_mean
        + hidden * latent + latent                # to_log_variance
        + n_corr * (2 * wide) + 2 * wide          # film1
        + n_corr * (2 * hidden) + 2 * hidden      # film2
    )
    assert sum(p.numel() for p in encoder.parameters()) == expected


def test_no_correction_variables_means_no_film_layers():
    encoder = CountEncoder(N_GENES, n_corrections=0, hidden_dim=HIDDEN, latent_dim=LATENT)
    assert encoder.film1 is None and encoder.film2 is None


# --------------------------------------------------------------------------
# Shapes.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_corrections", [0, 1, 3])
def test_count_encoder_shapes(n_corrections):
    encoder = CountEncoder(N_GENES, n_corrections, HIDDEN, LATENT)
    counts = torch.rand(N_SAMPLES, N_GENES)
    k = corrections(n_corrections) if n_corrections else None

    mean, log_variance = encoder(counts, k)
    assert mean.shape == (N_SAMPLES, LATENT)
    assert log_variance.shape == (N_SAMPLES, LATENT)


@pytest.mark.parametrize("n_corrections", [0, 1, 3])
def test_count_decoder_shapes(n_corrections):
    decoder = CountDecoder(LATENT, n_corrections, HIDDEN)
    latent = torch.rand(N_SAMPLES, LATENT)
    k = corrections(n_corrections) if n_corrections else None
    assert decoder(latent, k).shape == (N_SAMPLES, HIDDEN)


@pytest.mark.parametrize("n_corrections", [0, 2])
def test_library_encoder_and_decoder_shapes(n_corrections):
    encoder = LibrarySizeEncoder(n_corrections, HIDDEN, LATENT)
    decoder = LibrarySizeDecoder(LATENT, n_corrections, HIDDEN)
    k = corrections(n_corrections) if n_corrections else None

    log_library = torch.rand(N_SAMPLES) * 5 + 5
    mean, log_variance = encoder(log_library, k)
    assert mean.shape == (N_SAMPLES, LATENT)
    assert log_variance.shape == (N_SAMPLES, LATENT)
    assert decoder(mean, k).shape == (N_SAMPLES, 1)


def test_film_scales_and_shifts_every_feature():
    film = FiLM(n_conditions=2, n_features=6)
    features = torch.ones(4, 6)
    out = film(features, torch.zeros(4, 2))
    assert out.shape == (4, 6)


def test_film_output_depends_on_the_conditioning():
    torch.manual_seed(0)
    film = FiLM(n_conditions=1, n_features=5)
    features = torch.ones(3, 5)
    a = film(features, torch.zeros(3, 1))
    b = film(features, torch.ones(3, 1))
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------
# Missing correction variables are reported, not silently skipped.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "call"),
    [
        (lambda: CountEncoder(N_GENES, 2, HIDDEN, LATENT),
         lambda m: m(torch.rand(N_SAMPLES, N_GENES))),
        (lambda: CountDecoder(LATENT, 2, HIDDEN),
         lambda m: m(torch.rand(N_SAMPLES, LATENT))),
        (lambda: LibrarySizeEncoder(2, HIDDEN, LATENT),
         lambda m: m(torch.rand(N_SAMPLES))),
        (lambda: LibrarySizeDecoder(LATENT, 2, HIDDEN),
         lambda m: m(torch.rand(N_SAMPLES, LATENT))),
    ],
)
def test_omitting_expected_corrections_is_an_error(make, call):
    with pytest.raises(ValueError, match="correction"):
        call(make())


# --------------------------------------------------------------------------
# Gradients reach everything, so nothing is accidentally detached.
# --------------------------------------------------------------------------


def test_gradients_reach_every_count_encoder_parameter():
    encoder = CountEncoder(N_GENES, 1, HIDDEN, LATENT)
    mean, log_variance = encoder(torch.rand(N_SAMPLES, N_GENES), corrections(1))
    (mean.sum() + log_variance.sum()).backward()

    without = [n for n, p in encoder.named_parameters() if p.grad is None]
    assert without == []


def test_gradients_reach_every_library_encoder_parameter():
    encoder = LibrarySizeEncoder(1, HIDDEN, LATENT)
    mean, log_variance = encoder(torch.rand(N_SAMPLES) * 5 + 5, corrections(1))
    (mean.sum() + log_variance.sum()).backward()
    assert [n for n, p in encoder.named_parameters() if p.grad is None] == []


# --------------------------------------------------------------------------
# The seams.
#
# Two places where version 0.1 made a choice we doubt but have not measured.
# They are overridable methods rather than settings, so an experiment can vary
# exactly one of them without copying the class. See
# decisions/2026-10-15_odd_choices_kept_pending_measurement.md.
# --------------------------------------------------------------------------


def test_the_latent_projection_applies_no_activation():
    """Version 0.1 rectified both. exp/01 found no benefit, so it is gone.

    Decided in decisions/2026-10-15_exp01_latent_rectifier_removed.md. If this
    starts failing, something has reintroduced an activation on the parameters
    of a distribution, which wants a reason.
    """
    torch.manual_seed(0)
    encoder = CountEncoder(N_GENES, 0, HIDDEN, LATENT)
    encoder.eval()

    hidden = torch.randn(N_SAMPLES, HIDDEN) * 5
    mean, log_variance = encoder.project_to_latent(hidden)

    assert (mean < 0).any(), "the fixture produced no negative values to check"
    torch.testing.assert_close(mean, encoder.to_mean(hidden))
    torch.testing.assert_close(log_variance, encoder.to_log_variance(hidden))


def test_overriding_the_latent_projection_changes_the_result():
    """The seam still works, so the comparison can be repeated at full scale."""

    class Rectified(CountEncoder):
        def project_to_latent(self, hidden):
            return (
                self.activation(self.to_mean(hidden)),
                self.activation(self.to_log_variance(hidden)),
            )

    torch.manual_seed(0)
    plain = CountEncoder(N_GENES, 0, HIDDEN, LATENT)
    torch.manual_seed(0)
    varied = Rectified(N_GENES, 0, HIDDEN, LATENT)
    plain.eval()
    varied.eval()

    counts = torch.rand(N_SAMPLES, N_GENES)
    assert not torch.allclose(plain(counts)[0], varied(counts)[0])


def test_overriding_the_correction_seam_changes_the_result():
    class Doubled(CountEncoder):
        def combine_with_corrections(self, counts, corrections):
            return torch.cat((counts, corrections.float() * 2), dim=1)

    torch.manual_seed(0)
    plain = CountEncoder(N_GENES, 1, HIDDEN, LATENT)
    torch.manual_seed(0)
    varied = Doubled(N_GENES, 1, HIDDEN, LATENT)
    plain.eval()
    varied.eval()

    counts, k = torch.rand(N_SAMPLES, N_GENES), torch.ones(N_SAMPLES, 1)
    assert not torch.allclose(plain(counts, k)[0], varied(counts, k)[0])


def test_dropout_is_inactive_once_the_model_is_in_evaluation_mode():
    """Otherwise the same input gives a different answer every time it is scored."""
    torch.manual_seed(0)
    encoder = CountEncoder(N_GENES, 0, HIDDEN, LATENT)
    encoder.eval()
    counts = torch.rand(N_SAMPLES, N_GENES)
    first = encoder(counts)[0]
    second = encoder(counts)[0]
    torch.testing.assert_close(first, second)


# --------------------------------------------------------------------------
# The encoder is fed log(1 + counts).
#
# Defect (finding 49): both versions fed raw counts, which reach 2.4 million
# in the DCIS cohort, straight to the encoder's first layer. A learning-rate-
# sized change to one weight times an input that size moved the encoder's
# output a thousandfold in a single step: the loss spikes of issue #22.
# exp/10 measured the log scale against raw counts, six seeds, and the
# decision of 2026-10-28 made it the default.
# --------------------------------------------------------------------------


def test_the_encoder_sees_log_counts():
    """A count of a million reaches the first layer as about 14."""
    encoder = CountEncoder(N_GENES, 0, HIDDEN, LATENT)
    seen = encoder.scale_input(torch.tensor([[0.0, 1.0, 1e6]]))
    assert torch.allclose(seen, torch.tensor([[0.0, 0.6931, 13.8155]]), atol=1e-3)


def test_the_scaling_is_on_the_path_to_the_latent():
    """Feeding the first layer raw counts instead gives a different latent."""
    torch.manual_seed(0)
    encoder = CountEncoder(N_GENES, 0, HIDDEN, LATENT).eval()  # no dropout
    raw = torch.rand(N_SAMPLES, N_GENES) * 1e6
    scaled_mean, _ = encoder(raw, None)
    encoder.scale_input = lambda counts: counts
    raw_mean, _ = encoder(raw, None)
    assert not torch.allclose(scaled_mean, raw_mean)


def test_a_learning_rate_sized_step_cannot_move_the_latent_far():
    """One weight moved by the learning rate, on counts of a million, barely shifts the latent.

    On raw counts the same change shifts a pre-activation by 500.
    """
    torch.manual_seed(0)
    encoder = CountEncoder(N_GENES, 0, HIDDEN, LATENT).eval()  # no dropout
    raw = torch.rand(N_SAMPLES, N_GENES) * 1e6
    with torch.no_grad():
        before, _ = encoder(raw, None)
        encoder.layer1.weight[0, 0] += 5e-4
        after, _ = encoder(raw, None)
    assert float((after - before).abs().max()) < 0.1

