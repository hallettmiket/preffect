"""The single model: counts plus a sample-sample network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric", reason="needs the graph extra")

from preffect.config import (  # noqa: E402
    DataConfig,
    Likelihood,
    LossWeights,
    ModelConfig,
    ModelKind,
    PreffectConfig,
    RuntimeConfig,
    TrainConfig,
)
from preffect.data.graph import edge_survival, induce_subgraph  # noqa: E402
from preffect.data.tensors import TensorData  # noqa: E402
from preffect.errors import ConfigError, DataError  # noqa: E402
from preffect.graph_model import SingleModel  # noqa: E402
from preffect.loss import compute_loss  # noqa: E402
from preffect.model import SimpleModel  # noqa: E402
from preffect.training import train  # noqa: E402

N_GENES, N_SAMPLES = 18, 12


def ring(n: int = N_SAMPLES) -> torch.Tensor:
    pairs = [(i, (i + 1) % n) for i in range(n)]
    return torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).t()



def _dense_ring(n: int) -> torch.Tensor:
    """The same ring as ``ring()``, as the square matrix the objective scores."""
    network = torch.zeros(n, n)
    for i in range(n):
        network[i, (i + 1) % n] = 1.0
        network[(i + 1) % n, i] = 1.0
    return network


def build(**changes) -> SingleModel:
    config = ModelConfig(
        kind=ModelKind.SINGLE, hidden_dim=16, latent_dim=5, attention_heads=2, **changes
    )
    torch.manual_seed(0)
    return SingleModel(N_GENES, config)


def counts() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 60, (N_SAMPLES, N_GENES)).float()


def data_with_network() -> TensorData:
    x = counts()
    return TensorData(inputs=x.clone(), targets=x, edge_index=ring())


# --------------------------------------------------------------------------
# The model builds and runs.
# --------------------------------------------------------------------------


def test_the_pass_produces_both_halves():
    out = build()(counts(), None, ring())
    assert out.composition.shape == (N_SAMPLES, N_GENES)
    assert out.predicted_network.shape == (N_SAMPLES, N_SAMPLES)
    assert out.network_latent_mean.shape == (N_SAMPLES, 5)


def test_the_count_half_is_reachable_without_unwrapping():
    out = build()(counts(), None, ring())
    assert out.latent_mean.shape == out.counts_part.latent_mean.shape
    assert torch.equal(out.composition, out.counts_part.composition)


def test_the_two_latent_spaces_are_separate():
    """One summarises expression, the other network position."""
    out = build()(counts(), None, ring())
    assert not torch.allclose(out.latent_mean, out.network_latent_mean)


@pytest.mark.parametrize("likelihood", [Likelihood.NB, Likelihood.ZINB])
@pytest.mark.parametrize("infer_library", [False, True])
def test_gradients_reach_every_parameter(likelihood, infer_library):
    model = build(likelihood=likelihood, infer_library_size=infer_library)
    x = counts()
    out = model(x, None, ring())
    (-out.counts.log_prob(x).sum() + out.predicted_network.sum()).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


# --------------------------------------------------------------------------
# A request for a graph model is honoured or refused, never rewritten.
#
# Defect 2: version 0.1 silently turned a request for the single model into
# the simple model, then loaded the network and ignored it.
# --------------------------------------------------------------------------


def test_building_a_single_model_with_kind_simple_is_refused():
    with pytest.raises(ConfigError, match="finding 2"):
        SingleModel(N_GENES, ModelConfig(kind=ModelKind.SIMPLE, hidden_dim=16, latent_dim=5))


def test_running_without_a_network_is_refused():
    with pytest.raises(ConfigError, match="needs a sample-sample network"):
        build()(counts(), None, None)


def test_asking_for_a_network_that_is_not_there_is_an_error():
    import anndata as ad
    import pandas as pd

    from preffect.data.tensors import to_tensors

    adata = ad.AnnData(
        X=np.zeros((4, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(4)]),
    )
    with pytest.raises(DataError, match="no sample-sample network"):
        to_tensors(adata, with_network=True)


# --------------------------------------------------------------------------
# Batching the network.
# --------------------------------------------------------------------------


def test_selecting_a_batch_induces_the_subgraph():
    part = data_with_network().select(np.array([0, 1, 2]))
    assert part.has_network
    # On a ring, samples 0-1-2 keep two edges, in both directions.
    assert part.edge_index.shape[1] == 4
    assert int(part.edge_index.max()) < 3  # renumbered locally


def test_a_batch_of_strangers_keeps_no_edges():
    part = data_with_network().select(np.array([0, 5, 9]))
    assert part.edge_index.shape[1] == 0


def test_the_adjacency_matrix_matches_the_edges():
    part = data_with_network().select(np.array([0, 1, 2]))
    matrix = part.adjacency_matrix()
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 4
    assert matrix[0, 1] == 1 and matrix[1, 2] == 1 and matrix[0, 2] == 0


def test_edge_survival_reports_the_share_kept():
    """Finding 22: the share is the batch size over the sample count."""
    edges = ring(100)
    rng = np.random.default_rng(0)
    from preffect.data.batching import batch_indices

    batches = batch_indices(100, 50, rng=rng)
    survival = edge_survival(edges, batches, 100)
    assert survival.total_edges == 100
    assert 0.3 < survival.fraction < 0.7  # about half, for half-sized batches
    assert "usable" in str(survival)


def test_induce_subgraph_drops_edges_leaving_the_batch():
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    kept = induce_subgraph(edges, torch.tensor([0, 1, 2]), 4)
    assert kept.shape[1] == 2  # 0-1 and 1-2 survive, 2-3 does not


# --------------------------------------------------------------------------
# The objective gains two terms, and only for this model.
# --------------------------------------------------------------------------


def test_the_network_terms_are_present_for_a_graph_model():
    part = data_with_network()
    out = build()(part.inputs, None, part.edge_index)
    loss = compute_loss(out, part.targets, LossWeights(), network=part.adjacency_matrix())
    assert {"adjacency_recon", "adjacency_kl"} <= {t.name for t in loss.terms}


def test_the_network_terms_are_absent_for_the_simple_model():
    """Absent, not zero: a term that does not apply is not in the divisor."""
    torch.manual_seed(0)
    model = SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=5))
    x = counts()
    loss = compute_loss(model(x), x, LossWeights())
    names = {t.name for t in loss.terms}
    assert "adjacency_recon" not in names and "adjacency_kl" not in names


def test_the_graph_model_has_a_larger_divisor():
    part = data_with_network()
    graph_loss = compute_loss(
        build()(part.inputs, None, part.edge_index),
        part.targets,
        LossWeights(),
        network=part.adjacency_matrix(),
    )
    torch.manual_seed(0)
    simple = SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=5))
    simple_loss = compute_loss(simple(part.inputs), part.targets, LossWeights())
    assert graph_loss.weight_total > simple_loss.weight_total


# --------------------------------------------------------------------------
# It trains.
# --------------------------------------------------------------------------


def test_the_single_model_trains():
    config = PreffectConfig(
        data=DataConfig(train_path=Path("/nowhere")),
        model=ModelConfig(
            kind=ModelKind.SINGLE, hidden_dim=16, latent_dim=5, attention_heads=2
        ),
        loss=LossWeights(),
        train=TrainConfig(epochs=12, batch_size=6, learning_rate=1e-2),
        runtime=RuntimeConfig(seed=0),
    )
    torch.manual_seed(0)
    model = SingleModel(N_GENES, config.model)
    history = train(model, data_with_network(), config)

    assert len(history) == 12
    assert history.train_losses[-1] < history.train_losses[0]
    assert "adjacency_recon" in history.epochs[-1].train.mean_terms


# --------------------------------------------------------------------------
# The real objective must reach the network decoder.
#
# Finding 31: swapping the adjacency term's arguments and detaching its input
# passed the whole suite. test_gradients_reach_every_parameter did not notice,
# because it builds its own surrogate objective
# (-log_prob(x).sum() + predicted_network.sum()) rather than calling
# compute_loss. That tests the model is differentiable, which is a weaker
# claim: every gradient path from the objective to the network decoder can be
# cut with the suite green.
# --------------------------------------------------------------------------


def test_the_real_objective_reaches_the_encoder_through_the_network_term():
    """The adjacency decoder has no parameters of its own, so the question is
    whether the network term changes what the encoder learns.

    Compare the encoder's gradient under two very different weights on the
    network term. If the term is detached, scored against the wrong argument,
    or simply absent, changing its weight changes nothing. A weight of zero
    would be the cleaner contrast, but the settings refuse one, so this uses
    the smallest and largest weights they accept.
    """
    from preffect.config import LossWeights
    from preffect.loss import compute_loss

    x, edges, network = counts(), ring(), _dense_ring(N_SAMPLES)

    def encoder_gradient(adjacency_weight: float) -> torch.Tensor:
        model = build()
        weights = LossWeights(adjacency_recon=(adjacency_weight,))
        torch.manual_seed(5)
        loss = compute_loss(
            model(x, None, edges), x, weights=weights, network=network
        )
        loss.total.backward()
        return model.encoder.to_mean.weight.grad.clone()

    light = encoder_gradient(0.01)
    heavy = encoder_gradient(100.0)

    assert not torch.allclose(light, heavy), (
        "weighting the network term ten thousand times more heavily did not "
        "change the encoder's gradient, so nothing in the objective is "
        "learning from the sample-sample network"
    )


def test_the_adjacency_term_is_in_the_objective_at_all():
    """Named, present, and carrying a gradient. Not merely computed."""
    from preffect.config import LossWeights
    from preffect.loss import compute_loss

    loss = compute_loss(
        build()(counts(), None, ring()), counts(),
        weights=LossWeights(), network=_dense_ring(N_SAMPLES),
    )
    names = {term.name for term in loss.terms}
    assert "adjacency_recon" in names, f"terms were {sorted(names)}"

    term = next(t for t in loss.terms if t.name == "adjacency_recon")
    assert term.value.requires_grad, "the adjacency term was detached"
