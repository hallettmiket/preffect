"""The graph encoder and the adjacency decoder."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_geometric", reason="needs the graph extra")

from preffect.graph_layers import AdjacencyDecoder, GraphCountEncoder  # noqa: E402

N_GENES, N_SAMPLES = 24, 10
HIDDEN, LATENT, HEADS = 12, 5, 4


def ring_edges(n: int = N_SAMPLES) -> torch.Tensor:
    """A ring, so every sample has two neighbours and nothing is isolated."""
    pairs = [(i, (i + 1) % n) for i in range(n)]
    both = pairs + [(b, a) for a, b in pairs]
    return torch.tensor(both, dtype=torch.long).t()


def counts() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(N_SAMPLES, N_GENES) * 50


def corrections(n: int = 1) -> torch.Tensor:
    return torch.randint(0, 2, (N_SAMPLES, n)).float()


# --------------------------------------------------------------------------
# Shapes.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_corrections", [0, 1, 3])
def test_encoder_shapes(n_corrections):
    encoder = GraphCountEncoder(N_GENES, n_corrections, HIDDEN, LATENT, HEADS)
    k = corrections(n_corrections) if n_corrections else None
    mean, log_variance = encoder(counts(), ring_edges(), k)
    assert mean.shape == (N_SAMPLES, LATENT)
    assert log_variance.shape == (N_SAMPLES, LATENT)


@pytest.mark.parametrize("n_corrections", [0, 2])
def test_adjacency_decoder_shapes(n_corrections):
    decoder = AdjacencyDecoder(LATENT, n_corrections, HIDDEN)
    k = corrections(n_corrections) if n_corrections else None
    predicted = decoder(torch.rand(N_SAMPLES, LATENT), k)
    assert predicted.shape == (N_SAMPLES, N_SAMPLES)


def test_predicted_edges_are_probabilities():
    decoder = AdjacencyDecoder(LATENT, 0, HIDDEN)
    predicted = decoder(torch.randn(N_SAMPLES, LATENT) * 5)
    assert (predicted >= 0).all() and (predicted <= 1).all()


def test_predicted_network_is_symmetric():
    """An inner product is symmetric, and so is a resemblance between samples."""
    decoder = AdjacencyDecoder(LATENT, 0, HIDDEN)
    decoder.eval()
    predicted = decoder(torch.randn(N_SAMPLES, LATENT))
    torch.testing.assert_close(predicted, predicted.transpose(0, 1))


# --------------------------------------------------------------------------
# The decoder does not depend on the batch size.
#
# Defect: version 0.1's adjacency decoder was a linear layer of width equal to
# the mini-batch size, so its parameter count followed a training setting,
# 7,400 at a batch of 50 and 15,050 at a batch of 200, and it could only ever
# process batches of exactly that size. That is why the last short batch had
# to be discarded, which is a defect we had already fixed.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [1, 7, 50, 213])
def test_any_batch_size_works(n_samples):
    decoder = AdjacencyDecoder(LATENT, 0, HIDDEN)
    predicted = decoder(torch.rand(n_samples, LATENT))
    assert predicted.shape == (n_samples, n_samples)


def test_the_parameter_count_does_not_depend_on_the_batch_size():
    decoder = AdjacencyDecoder(LATENT, 0, HIDDEN)
    before = sum(p.numel() for p in decoder.parameters())
    decoder(torch.rand(500, LATENT))  # a much larger batch
    assert sum(p.numel() for p in decoder.parameters()) == before


def test_identical_samples_get_identical_predicted_edges():
    """Two samples with the same latent values must be indistinguishable.

    Note what is *not* asserted: that the most similar pair always scores
    highest. An inner product conflates similarity with magnitude, so a sample
    with a large representation can score highly against everything. That is
    the conventional graph autoencoder decoder and it is deliberate, since the
    magnitude is how the model says "this sample connects to many things".
    """
    decoder = AdjacencyDecoder(LATENT, 0, HIDDEN)
    decoder.eval()
    latent = torch.randn(4, LATENT)
    latent[1] = latent[0]  # a duplicate of sample 0

    predicted = decoder(latent)
    torch.testing.assert_close(predicted[0], predicted[1])
    torch.testing.assert_close(predicted[:, 0], predicted[:, 1])


def test_the_decoder_can_learn_a_network():
    """The real test: can it reproduce a network it is shown?"""
    torch.manual_seed(0)
    decoder = AdjacencyDecoder(LATENT, 0, HIDDEN)
    latent = torch.randn(N_SAMPLES, LATENT)

    target = torch.zeros(N_SAMPLES, N_SAMPLES)
    for i in range(N_SAMPLES):  # the same ring the encoder tests use
        target[i, (i + 1) % N_SAMPLES] = 1.0
        target[(i + 1) % N_SAMPLES, i] = 1.0

    loss_fn = torch.nn.BCELoss()
    before = loss_fn(decoder(latent).clamp(1e-6, 1 - 1e-6), target)

    optimiser = torch.optim.Adam(decoder.parameters(), lr=1e-2)
    for _ in range(200):
        optimiser.zero_grad()
        loss_fn(decoder(latent).clamp(1e-6, 1 - 1e-6), target).backward()
        optimiser.step()

    decoder.eval()
    after = loss_fn(decoder(latent).clamp(1e-6, 1 - 1e-6), target)
    assert after < before


# --------------------------------------------------------------------------
# The network is actually used.
# --------------------------------------------------------------------------


def test_the_edges_change_the_answer():
    """If they did not, the graph models would be the simple model in disguise."""
    encoder = GraphCountEncoder(N_GENES, 0, HIDDEN, LATENT, HEADS)
    encoder.eval()
    x = counts()

    no_edges = torch.zeros(2, 0, dtype=torch.long)
    with_edges = encoder(x, ring_edges())[0]
    without = encoder(x, no_edges)[0]
    assert not torch.allclose(with_edges, without)


def test_a_sample_is_influenced_by_its_neighbours():
    """Changing a neighbour's counts should move this sample's representation."""
    encoder = GraphCountEncoder(N_GENES, 0, HIDDEN, LATENT, HEADS)
    encoder.eval()
    edges = ring_edges()

    x = counts()
    before = encoder(x, edges)[0][0]

    changed = x.clone()
    changed[1] += 500.0  # sample 1 is a neighbour of sample 0 on the ring
    after = encoder(changed, edges)[0][0]

    assert not torch.allclose(before, after)


def test_an_empty_network_still_runs():
    """A cohort where nothing resembles anything should not crash."""
    encoder = GraphCountEncoder(N_GENES, 0, HIDDEN, LATENT, HEADS)
    mean, _ = encoder(counts(), torch.zeros(2, 0, dtype=torch.long))
    assert mean.shape == (N_SAMPLES, LATENT)
    assert torch.isfinite(mean).all()


# --------------------------------------------------------------------------
# Latent projection and gradients.
# --------------------------------------------------------------------------


def test_the_latent_projection_applies_no_activation():
    """Matching the simple model after exp/01."""
    encoder = GraphCountEncoder(N_GENES, 0, HIDDEN, LATENT, HEADS)
    encoder.eval()
    hidden = torch.randn(N_SAMPLES, HIDDEN) * 5
    mean, log_variance = encoder.project_to_latent(hidden, None)

    assert (mean < 0).any()
    torch.testing.assert_close(mean, encoder.to_mean(hidden))
    torch.testing.assert_close(log_variance, encoder.to_log_variance(hidden))


def test_gradients_reach_every_encoder_parameter():
    encoder = GraphCountEncoder(N_GENES, 1, HIDDEN, LATENT, HEADS)
    mean, log_variance = encoder(counts(), ring_edges(), corrections(1))
    (mean.sum() + log_variance.sum()).backward()
    assert [n for n, p in encoder.named_parameters() if p.grad is None] == []


def test_gradients_reach_every_decoder_parameter():
    decoder = AdjacencyDecoder(LATENT, 1, HIDDEN)
    decoder(torch.rand(N_SAMPLES, LATENT), corrections(1)).sum().backward()
    assert [n for n, p in decoder.named_parameters() if p.grad is None] == []


def test_omitting_expected_corrections_is_an_error():
    encoder = GraphCountEncoder(N_GENES, 2, HIDDEN, LATENT, HEADS)
    with pytest.raises(ValueError, match="correction"):
        encoder(counts(), ring_edges())

    decoder = AdjacencyDecoder(LATENT, 2, HIDDEN)
    with pytest.raises(ValueError, match="correction"):
        decoder(torch.rand(N_SAMPLES, LATENT))


# --------------------------------------------------------------------------
# The upstream deprecation warning does not reach the user.
#
# torch-geometric 2.8.1 calls torch.jit.script while being imported and
# PyTorch 2.14 warns that it is deprecated, so every graph-model run opened
# with a FutureWarning about a library the user had not heard of. The filter
# in require_graph_support is scoped to that message and that import. This
# runs the import in a fresh interpreter with FutureWarning turned into an
# error, because in this process torch_geometric is already imported and the
# warning would not fire again whatever the filter did.
# --------------------------------------------------------------------------


def test_importing_the_graph_layer_raises_no_future_warning():
    import subprocess
    import sys

    pytest.importorskip("torch_geometric", reason="needs the graph extra")
    result = subprocess.run(
        [
            sys.executable, "-W", "error::FutureWarning", "-c",
            "from preffect.graph_layers import require_graph_support; "
            "layer = require_graph_support(); print(layer.__name__)",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "GATv2Conv" in result.stdout


# --------------------------------------------------------------------------
# The graph encoder is fed log(1 + counts), as the simple one is.
#
# Finding 49 and the decision of 2026-10-28. Measured on the simple model
# (exp/10); the graph models got the same scaling on the same reasoning, and
# issue #27 is the measurement they are owed.
# --------------------------------------------------------------------------


def test_the_graph_encoder_sees_log_counts():
    """A count of a million reaches the attention layer as about 14."""
    encoder = GraphCountEncoder(N_GENES, 0, HIDDEN, LATENT, HEADS)
    seen = encoder.scale_input(torch.tensor([[0.0, 1.0, 1e6]]))
    assert torch.allclose(seen, torch.tensor([[0.0, 0.6931, 13.8155]]), atol=1e-3)


def test_the_graph_scaling_is_on_the_path_to_the_latent():
    """Feeding the attention layer raw counts instead gives a different latent."""
    torch.manual_seed(0)
    encoder = GraphCountEncoder(N_GENES, 0, HIDDEN, LATENT, HEADS).eval()  # no dropout
    raw = torch.rand(N_SAMPLES, N_GENES) * 1e6
    scaled_mean, _ = encoder(raw, ring_edges(), None)
    encoder.scale_input = lambda counts: counts
    raw_mean, _ = encoder(raw, ring_edges(), None)
    assert not torch.allclose(scaled_mean, raw_mean)

