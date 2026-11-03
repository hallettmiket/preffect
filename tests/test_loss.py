"""The objective.

The first group is the one this rewrite exists for: the numerator and the
divisor are built from one list, so they cannot disagree about which terms are
in play. Version 0.1 kept two hand-written lists and they did disagree.
"""

from __future__ import annotations

import pytest
import torch

from preffect.config import Likelihood, LossDelays, LossWeights, ModelConfig
from preffect.errors import ConfigError
from preffect.loss import compute_loss, gaussian_kl_divergence
from preffect.model import SimpleModel

N_GENES, N_SAMPLES = 25, 10


def build(**changes) -> SimpleModel:
    return SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=6, **changes))


def counts() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 80, (N_SAMPLES, N_GENES)).float()


def run(model=None, **loss_kwargs):
    model = model or build()
    x = counts()
    torch.manual_seed(1)
    return compute_loss(model(x), x, **{"weights": LossWeights(), **loss_kwargs}), x


# --------------------------------------------------------------------------
# One list, so the numerator and the divisor cannot disagree.
#
# Defect: version 0.1 computed the library reconstruction term, weighted it,
# logged it, and left it out of the sum, while its weight of 100 stayed in
# the divisor. Separately, a simple model's divisor counted 100.1 of weight
# for adjacency terms it does not have.
# --------------------------------------------------------------------------


def test_the_divisor_is_the_weights_of_the_terms_present():
    """Against the settings, not against the Loss object's own arithmetic.

    **This test used to be ``X == X``.** It asserted
    ``loss.weight_total == sum(t.weight for t in loss.terms)``, and
    ``Loss.weight_total`` is *defined* as that sum, so it could not fail. An
    audit found it by inflating the divisor by 201.2 and watching this test,
    the one named for the property, stay green.

    That is the same shape as the defect this repository was started over: a
    check that cannot fail. See finding 31.
    """
    loss, _ = run()

    # Built from the settings, with no reference to the Loss object. These are
    # the only two terms a simple model with an observed library size has.
    weights = LossWeights()
    expected = weights.expression_recon + weights.expression_kl[0]
    assert loss.weight_total == pytest.approx(expected)

    # And the names are the ones a simple model should have, so a term that
    # silently appeared or vanished would show here rather than cancelling.
    assert {t.name for t in loss.terms} == {"expression_recon", "expression_kl"}


def test_every_term_present_contributes_to_the_total():
    """Rebuild the total by hand from the terms and get the same number."""
    loss, _ = run()
    by_hand = sum(t.weight * t.value for t in loss.terms) / loss.weight_total
    torch.testing.assert_close(loss.total, by_hand)


def test_a_term_that_does_not_apply_is_absent_from_both():
    """Not present-and-zero, which is what inflated version 0.1's divisor."""
    without, _ = run(build(infer_library_size=False))
    with_library, _ = run(build(infer_library_size=True))

    names_without = {t.name for t in without.terms}
    names_with = {t.name for t in with_library.terms}

    assert "library_recon" not in names_without
    assert "library_kl" not in names_without
    assert {"library_recon", "library_kl"} <= names_with
    # and the divisor moves with them, rather than counting absent terms
    assert with_library.weight_total > without.weight_total


def test_the_library_term_reaches_the_total_when_it_is_present():
    """The defect this project started from: computed, reported, not optimised."""
    loss, _ = run(build(infer_library_size=True))
    library = next(t for t in loss.terms if t.name == "library_recon")

    without_it = sum(
        t.weight * t.value for t in loss.terms if t.name != "library_recon"
    ) / loss.weight_total

    assert library.value.item() != 0
    assert not torch.isclose(loss.total, without_it)


def test_a_delayed_term_is_in_neither_the_total_nor_the_divisor():
    delays = LossDelays(expression_kl=5)
    model = build()
    x = counts()

    early = compute_loss(model(x), x, LossWeights(), delays, epoch=1)
    late = compute_loss(model(x), x, LossWeights(), delays, epoch=5)

    assert "expression_kl" not in {t.name for t in early.terms}
    assert "expression_kl" in {t.name for t in late.terms}
    assert late.weight_total > early.weight_total


def test_reporting_uses_the_same_terms_as_the_optimisation():
    """Version 0.1's printed table and its objective drifted apart."""
    loss, _ = run()
    assert set(loss.named_values()) == {t.name for t in loss.terms}
    assert set(loss.named_weighted()) == {t.name for t in loss.terms}


# --------------------------------------------------------------------------
# The terms mean what they say.
# --------------------------------------------------------------------------


def test_kl_divergence_is_zero_for_a_standard_gaussian():
    mean = torch.zeros(4, 6)
    log_variance = torch.zeros(4, 6)  # variance of one
    torch.testing.assert_close(
        gaussian_kl_divergence(mean, log_variance), torch.zeros(4)
    )


def test_kl_divergence_grows_as_the_latent_drifts():
    log_variance = torch.zeros(1, 6)
    near = gaussian_kl_divergence(torch.full((1, 6), 0.1), log_variance)
    far = gaussian_kl_divergence(torch.full((1, 6), 3.0), log_variance)
    assert far.item() > near.item() > 0


def test_kl_divergence_is_never_negative():
    torch.manual_seed(0)
    values = gaussian_kl_divergence(torch.randn(50, 6), torch.randn(50, 6))
    assert (values >= -1e-6).all()


def test_reconstruction_falls_when_the_fit_improves():
    """A model fitted to the data should score better than a fresh one."""
    model = build()
    x = counts()
    torch.manual_seed(1)
    before = compute_loss(model(x), x, LossWeights())

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(40):
        optimiser.zero_grad()
        compute_loss(model(x), x, LossWeights()).total.backward()
        optimiser.step()

    torch.manual_seed(1)
    after = compute_loss(model(x), x, LossWeights())
    assert after.named_values()["expression_recon"] < before.named_values()[
        "expression_recon"
    ]


def test_the_batch_penalty_is_lower_when_batches_are_already_tight():
    from preffect.loss import _batch_centroid_penalty

    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    tight = torch.tensor([[0.0, 0.0]] * 3 + [[5.0, 5.0]] * 3)
    loose = torch.tensor(
        [[0.0, 0.0], [4.0, 4.0], [-4.0, -4.0], [5.0, 5.0], [9.0, 9.0], [1.0, 1.0]]
    )
    assert _batch_centroid_penalty(tight, labels) < _batch_centroid_penalty(loose, labels)


# --------------------------------------------------------------------------
# Training.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("infer_library", [False, True])
@pytest.mark.parametrize("likelihood", [Likelihood.NB, Likelihood.ZINB])
def test_the_total_carries_gradient_to_every_parameter(infer_library, likelihood):
    model = build(infer_library_size=infer_library, likelihood=likelihood)
    x = counts()
    compute_loss(model(x), x, LossWeights()).total.backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_the_total_is_finite_on_extreme_counts():
    x = counts()
    x[0, 0] = 1e8
    loss, _ = run(build(), )
    assert torch.isfinite(loss.total)


def test_turning_normalisation_off_gives_the_unweighted_sum():
    model = build()
    x = counts()
    torch.manual_seed(1)
    plain = compute_loss(model(x), x, LossWeights(normalise_by_weight_sum=False))
    by_hand = sum(t.weight * t.value for t in plain.terms)
    torch.testing.assert_close(plain.total, by_hand)
    assert not plain.normalised


# --------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------


def test_an_epoch_before_every_delay_is_refused_rather_than_silently_empty():
    delays = LossDelays(expression_recon=10, expression_kl=10)
    model = build()
    x = counts()
    with pytest.raises(ConfigError, match="no loss term is active"):
        compute_loss(model(x), x, LossWeights(), delays, epoch=1)


def test_epochs_count_from_one():
    model = build()
    x = counts()
    with pytest.raises(ConfigError, match="count from 1"):
        compute_loss(model(x), x, LossWeights(), epoch=0)


def test_a_delay_before_epoch_one_is_refused():
    with pytest.raises(ConfigError, match="count from 1"):
        LossDelays(expression_kl=0)


def test_an_unknown_term_name_is_refused():
    with pytest.raises(ConfigError, match="no delay is defined"):
        LossDelays().active_at("not_a_term", 1)


def test_mismatched_batch_labels_are_refused():
    from preffect.loss import _batch_centroid_penalty

    with pytest.raises(ConfigError, match="3 batch labels for 6 samples"):
        _batch_centroid_penalty(torch.zeros(6, 2), torch.tensor([0, 1, 0]))


# --------------------------------------------------------------------------
# The network term's three modes mean what they say.
#
# Issue #13: on a network one per cent dense an unweighted cross-entropy is
# almost entirely "no edge here". Each mode is checked on a six-sample
# network with two edges whose value can be worked out by hand, and 'none' is
# checked to be the unweighted mean it always was, bit for bit. The sampled
# term that lost exp/17's comparison lives in that experiment and is checked
# in tests/test_experiments.py.
# --------------------------------------------------------------------------


def two_edge_network(n: int = 6) -> torch.Tensor:
    """Edges 0-1 and 2-3: four linked entries, 26 absent ones off the diagonal."""
    observed = torch.zeros(n, n)
    for a, b in [(0, 1), (2, 3)]:
        observed[a, b] = observed[b, a] = 1.0
    return observed


def a_prediction(n: int = 6) -> torch.Tensor:
    """Probabilities with nothing special about them, symmetric."""
    torch.manual_seed(3)
    p = torch.rand(n, n) * 0.8 + 0.1
    return (p + p.t()) / 2


def per_entry_cross_entropy(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    return -(observed * predicted.log() + (1 - observed) * (1 - predicted).log())


def test_none_is_the_unweighted_mean_over_every_entry_bit_for_bit():
    """Including the diagonal, exactly as the term was computed before the
    setting existed, so every result produced under it stands."""
    from preffect.loss import network_term

    predicted, observed = a_prediction(), two_edge_network()
    before = torch.nn.functional.binary_cross_entropy(predicted.clamp(1e-7, 1 - 1e-7), observed)
    assert torch.equal(network_term(predicted, observed, "none"), before)
    assert torch.equal(network_term(predicted, observed), before)


def test_balanced_gives_the_linked_and_the_absent_pairs_half_the_weight_each():
    """By hand: 4 linked entries at weight 26/4 each, 26 absent entries at 1,
    the diagonal left out, divided by the total weight of 52."""
    from preffect.loss import network_term

    predicted, observed = a_prediction(), two_edge_network()
    entries = per_entry_cross_entropy(predicted, observed)
    off = ~torch.eye(6, dtype=torch.bool)
    linked = entries[(observed > 0) & off]
    absent = entries[(observed == 0) & off]
    assert len(linked) == 4 and len(absent) == 26
    by_hand = ((26 / 4) * linked.sum() + absent.sum()) / 52
    torch.testing.assert_close(network_term(predicted, observed, "balanced"), by_hand)
    # Which is the same as the two kinds' means averaged with equal weight.
    torch.testing.assert_close(
        network_term(predicted, observed, "balanced"), (linked.mean() + absent.mean()) / 2
    )


def test_the_two_modes_differ_on_the_same_network():
    from preffect.loss import network_term

    predicted, observed = a_prediction(), two_edge_network()
    values = {m: float(network_term(predicted, observed, m)) for m in ("none", "balanced")}
    assert values["none"] != values["balanced"], values


@pytest.mark.parametrize("mode", ["none", "balanced"])
def test_a_flat_prediction_of_one_half_scores_log_two_in_every_mode(mode):
    """The scales are comparable: the collapsed decoder reads the same number
    whichever mode is in force."""
    from preffect.loss import network_term

    value = network_term(torch.full((6, 6), 0.5), two_edge_network(), mode)
    torch.testing.assert_close(value, torch.tensor(float(torch.log(torch.tensor(2.0)))))


def test_balanced_leaves_the_diagonal_out():
    """The decoder's diagonal is never below one half, so scoring it against
    zero can only pull towards predicting one half everywhere."""
    from preffect.loss import network_term

    predicted, observed = a_prediction(), two_edge_network()
    reference = network_term(predicted, observed, "balanced")
    with_diagonal = predicted.clone()
    with_diagonal.fill_diagonal_(0.99)
    assert torch.equal(network_term(with_diagonal, observed, "balanced"), reference)


def test_a_batch_with_no_edge_scores_its_absent_pairs_plainly():
    """Nothing to balance against, so the off-diagonal mean, not a NaN."""
    from preffect.loss import network_term

    predicted = a_prediction()
    observed = torch.zeros(6, 6)
    value = network_term(predicted, observed, "balanced")
    off = ~torch.eye(6, dtype=torch.bool)
    torch.testing.assert_close(value, per_entry_cross_entropy(predicted, observed)[off].mean())


def test_an_unknown_mode_is_refused_by_name():
    from preffect.loss import network_term

    for mode in ("weighted", "sampled"):
        with pytest.raises(ConfigError, match="edge_balance must be one of"):
            network_term(a_prediction(), two_edge_network(), mode)


def test_the_mode_reaches_every_tissue_of_the_objective():
    """A graph model's objective under 'balanced' differs from under 'none' on
    every network term it has, so the setting is not read for one tissue and
    forgotten for the next. Both modes are named: the default is one of them,
    and a test that compared the default with itself would pass for nothing."""
    pytest.importorskip("torch_geometric", reason="needs the graph extra")
    from preffect.config import ModelKind
    from preffect.full_model import FullModel

    n_samples = 8
    config = ModelConfig(kind=ModelKind.FULL, hidden_dim=16, latent_dim=4, attention_heads=2)
    torch.manual_seed(0)
    model = FullModel(N_GENES, config, n_tissues=2).eval()
    torch.manual_seed(0)
    x = [torch.randint(0, 80, (n_samples, N_GENES)).float() for _ in range(2)]
    pairs = [(i, (i + 1) % n_samples) for i in range(n_samples)]
    edges = torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).t()
    networks = []
    for _ in range(2):
        dense = torch.zeros(n_samples, n_samples)
        dense[edges[0], edges[1]] = 1.0
        networks.append(dense)
    with torch.no_grad():
        torch.manual_seed(1)
        output = model(x, None, [edges, edges])
        plain = compute_loss(
            output, x[0], LossWeights(edge_balance="none").for_tissue_count(2), network=networks
        ).named_values()
        balanced = compute_loss(
            output, x[0], LossWeights(edge_balance="balanced").for_tissue_count(2),
            network=networks,
        ).named_values()
    for name in ("adjacency_recon_1", "adjacency_recon_2"):
        assert plain[name] != balanced[name], name
    assert plain["expression_recon"] == balanced["expression_recon"]


# --------------------------------------------------------------------------
# Leaving out rows that stand in for a missing tissue. Issue #14.
#
# Only the full model can be given such rows. The seam that leaves them out
# must therefore be invisible to the simple and single models: not nearly
# invisible, but equal to the last bit, because every experiment and decision
# record in this project was measured without it.
# --------------------------------------------------------------------------


def test_the_simple_model_is_scored_exactly_as_it_was():
    """No selection asked for, no selection applied, bit for bit."""
    model = build()
    plain, _ = run(model=model)
    explicit, _ = run(model=model, real_rows=None)
    assert plain.named_values() == explicit.named_values()
    assert float(plain.total.detach()) == float(explicit.total.detach())


def test_the_expression_term_is_the_plain_mean_when_nothing_is_left_out():
    model = build()
    loss, x = run(model=model)
    torch.manual_seed(1)
    expected = -model(x).counts.log_prob(x).sum(dim=-1).mean()
    assert loss.named_values()["expression_recon"] == pytest.approx(float(expected.detach()))


def test_leaving_rows_out_changes_only_the_terms_that_read_the_counts():
    model = build(infer_library_size=True)
    keep = torch.ones(N_SAMPLES, dtype=torch.bool)
    keep[-3:] = False
    over_all, _ = run(model=model)
    over_real, _ = run(model=model, real_rows=keep)
    all_values, real_values = over_all.named_values(), over_real.named_values()
    for name in ("expression_recon", "library_recon", "library_kl"):
        assert all_values[name] != real_values[name], name
    assert all_values["expression_kl"] == real_values["expression_kl"]


def test_leaving_every_row_out_is_refused_rather_than_averaged_over_nothing():
    """Version 0.1 averaged over nothing here and the loss stopped being a number."""
    with pytest.raises(ConfigError, match="every one of this batch"):
        run(real_rows=torch.zeros(N_SAMPLES, dtype=torch.bool))
