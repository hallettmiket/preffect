"""The count distributions.

Two kinds of check. First, that version 0.2 agrees numerically with version
0.1, so the rewrite did not quietly change the likelihood. Second, that the
distributions are correct in their own right, which version 0.1 never tested
and which would have caught a transcription error in either version.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from preffect.distributions import (
    NegativeBinomial,
    ZeroInflatedNegativeBinomial,
    log_prob_negative_binomial,
    log_prob_zero_inflated_negative_binomial,
)

GOLDEN = Path(__file__).parent / "data" / "version_0_1_distributions.json"


def load_cases():
    with open(GOLDEN) as fh:
        return json.load(fh)["cases"]


CASES = load_cases()
CASE_NAMES = [c["name"] for c in CASES]


def tensors(case):
    return (
        torch.tensor(case["x"]),
        torch.tensor(case["mu"]),
        torch.tensor(case["theta"]),
        torch.tensor(case["zero_logits"]),
    )


# --------------------------------------------------------------------------
# Agreement with version 0.1.
#
# The reference values were produced by version 0.1 under its own environment
# (Python 3.9, torch 2.5.1) and committed, so this runs without version 0.1
# present. Regenerate with tools/make_golden_distributions.py.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_NAMES)
def test_negative_binomial_matches_version_0_1(case):
    x, mu, theta, _ = tensors(case)
    expected = torch.tensor(case["log_prob_nb"])
    actual = log_prob_negative_binomial(x, mu, theta)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("case", CASES, ids=CASE_NAMES)
def test_zero_inflated_matches_version_0_1(case):
    x, mu, theta, zero_logits = tensors(case)
    expected = torch.tensor(case["log_prob_zinb"])
    actual = log_prob_zero_inflated_negative_binomial(x, mu, theta, zero_logits)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_the_reference_covers_zero_and_non_zero_counts():
    """Otherwise the agreement above would only exercise one branch."""
    for case in CASES:
        x = torch.tensor(case["x"])
        assert (x == 0).any(), f"{case['name']} has no zero counts"
        assert (x > 0).any(), f"{case['name']} has no non-zero counts"


# --------------------------------------------------------------------------
# Correct in their own right.
#
# A likelihood that is subtly wrong still trains, it just trains towards the
# wrong thing. Version 0.1 had no test of this kind.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mu", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("theta", [0.3, 1.0, 20.0])
def test_negative_binomial_probabilities_sum_to_one(mu, theta):
    """Summed over every possible count, a probability distribution gives 1."""
    counts = torch.arange(0, 2000).double()
    mu_t = torch.full_like(counts, mu, dtype=torch.double)
    theta_t = torch.full_like(counts, theta, dtype=torch.double)
    total = log_prob_negative_binomial(counts, mu_t, theta_t).exp().sum()
    assert total.item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("zero_logit", [-3.0, 0.0, 2.0])
def test_zero_inflated_probabilities_sum_to_one(zero_logit):
    counts = torch.arange(0, 2000).double()
    mu = torch.full_like(counts, 5.0, dtype=torch.double)
    theta = torch.full_like(counts, 1.5, dtype=torch.double)
    logits = torch.full_like(counts, zero_logit, dtype=torch.double)
    total = log_prob_zero_inflated_negative_binomial(counts, mu, theta, logits).exp().sum()
    assert total.item() == pytest.approx(1.0, abs=1e-6)


def test_mean_and_variance_match_the_distribution():
    """Checked against a large sample, not against the formula restated."""
    torch.manual_seed(0)
    mu = torch.full((1,), 30.0)
    theta = torch.full((1,), 2.0)
    nb = NegativeBinomial(mu=mu, theta=theta)

    draws = nb.sample((200_000,)).double()
    assert draws.mean().item() == pytest.approx(nb.mean.item(), rel=0.02)
    assert draws.var().item() == pytest.approx(nb.variance.item(), rel=0.05)


def test_variance_exceeds_the_mean():
    """The whole point of the negative binomial over the Poisson."""
    nb = NegativeBinomial(mu=torch.tensor([10.0]), theta=torch.tensor([2.0]))
    assert (nb.variance > nb.mean).all()


def test_large_theta_approaches_the_poisson():
    """As theta grows the extra spread vanishes and variance meets the mean."""
    mu = torch.tensor([10.0])
    spread = [
        NegativeBinomial(mu=mu, theta=torch.tensor([t])).variance.item()
        for t in (1.0, 100.0, 10_000.0)
    ]
    assert spread[0] > spread[1] > spread[2]
    assert spread[2] == pytest.approx(10.0, rel=0.01)


# --------------------------------------------------------------------------
# The zero-inflated form contains the plain one.
# --------------------------------------------------------------------------


def test_no_extra_zeros_reduces_to_the_negative_binomial():
    x = torch.arange(0, 20).float()
    mu = torch.full_like(x, 7.0)
    theta = torch.full_like(x, 1.5)
    never = torch.full_like(x, -30.0)  # log-odds of about zero probability

    plain = log_prob_negative_binomial(x, mu, theta)
    inflated = log_prob_zero_inflated_negative_binomial(x, mu, theta, never)
    torch.testing.assert_close(inflated, plain, rtol=1e-5, atol=1e-5)


def test_extra_zeros_raise_the_probability_of_a_zero():
    x = torch.zeros(1)
    mu, theta = torch.tensor([5.0]), torch.tensor([2.0])
    low = log_prob_zero_inflated_negative_binomial(x, mu, theta, torch.tensor([-5.0]))
    high = log_prob_zero_inflated_negative_binomial(x, mu, theta, torch.tensor([5.0]))
    assert high.item() > low.item()


def test_extra_zeros_lower_the_mean():
    mu, theta = torch.tensor([10.0]), torch.tensor([2.0])
    zinb = ZeroInflatedNegativeBinomial(mu, theta, zero_logits=torch.tensor([0.0]))
    assert zinb.mean.item() == pytest.approx(5.0)  # half the counts forced to zero


# --------------------------------------------------------------------------
# Shapes.
# --------------------------------------------------------------------------


def test_mismatched_parameter_shapes_are_refused():
    with pytest.raises(ValueError, match="have to match"):
        NegativeBinomial(mu=torch.ones(3, 4), theta=torch.ones(3, 5))
    with pytest.raises(ValueError, match="have to match"):
        ZeroInflatedNegativeBinomial(
            mu=torch.ones(3, 4), theta=torch.ones(3, 4), zero_logits=torch.ones(2, 4)
        )


def test_log_prob_keeps_the_shape_of_the_counts():
    nb = NegativeBinomial(mu=torch.ones(5, 7) * 3, theta=torch.ones(5, 7))
    assert nb.log_prob(torch.ones(5, 7)).shape == (5, 7)


def test_sampling_adds_the_requested_leading_shape():
    nb = NegativeBinomial(mu=torch.ones(5, 7) * 3, theta=torch.ones(5, 7))
    assert nb.sample((11,)).shape == (11, 5, 7)


def test_gradients_flow_to_both_parameters():
    """The model learns through these, so they have to be differentiable."""
    mu = torch.full((4,), 5.0, requires_grad=True)
    theta = torch.full((4,), 2.0, requires_grad=True)
    log_prob_negative_binomial(torch.arange(4).float(), mu, theta).sum().backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
