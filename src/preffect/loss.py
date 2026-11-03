"""The objective, assembled as a list of named terms.

The model is trained by making one number small. That number is a weighted
average of several terms, each measuring a different kind of error.

## Why a list

Version 0.1 built the objective as a hand-written sum of named variables, and
kept a second, separate hand-written list of weights to divide it by. The two
disagreed. The library size reconstruction term was computed, multiplied by its
weight, written to the log and to the printed table, and then left out of the
sum, while its weight of 100 per tissue stayed in the divisor. Separately, the
divisor for a simple model counted 100.1 of weight for adjacency terms a simple
model does not have.

Here the terms are collected into one list and both the numerator and the
divisor are computed from that list. A term in the list contributes to both. A
term not in the list contributes to neither. They cannot disagree about which
terms exist, because there is only one answer to that question.

This does not make every mistake impossible. A term can still compute the wrong
quantity, or be given a silly weight. Those are different mistakes, and tests
are what catch them.

## The terms

- **expression_recon**: how surprised the fitted distribution is by the counts
  actually observed. The main term. Negative log-likelihood, summed over genes
  and averaged over samples.
- **expression_kl**: how far the latent summary of each sample has drifted from
  a standard Gaussian. This is what keeps the latent space usable rather than
  letting the model memorise each sample separately.
- **library_recon**: how far the predicted library size is from the observed
  one. Present only when the library size is being inferred.
- **library_kl**: the same drift penalty for the library latent space.
- **batch_centroid**: an optional penalty pulling samples from the same batch
  together in the latent space.
- **adjacency_recon**: how well the sample-sample network is reproduced.
  Graph models only.
- **adjacency_kl**: the drift penalty for the network's own latent space.
  Graph models only.

## Rows padded in for a missing tissue count towards some terms and not others

The full model can be given a cohort where some patients lack one of the
tissues. The missing profile is padded in as a row of per-gene means, and
which terms that row belongs in is a decision version 0.1 made and version
0.2 inherited: it belongs in the latent and network terms and not in the
three terms that read the counts. ``compute_loss`` takes ``real_rows`` for
this, and a cohort with nothing padded passes None and is unaffected to the
last bit. :mod:`preffect.data.tissues` has the rules and the decision record
of 2026-11-02 has the measurement.

## The network term is balanced

These networks are sparse, 0.2 to 8 per cent of possible pairs, so a plain
mean over every pair is mostly "no edge here". Version 0.1
carried settings for weighting present against absent edges and left both at
one, and under that term the network half of the model learns nothing.
Measured in ``exp/17`` on synthetic data with a known network, six seeds: the
decoder settles on predicting no edge anywhere, the network latent collapses
onto its prior, and the term sits at ln 2, in every run. With the linked pairs
given the same total weight as the absent ones, the predicted network ranks
the true edges as well as a predictor that knows the sample groups, and the
expression half of the model is within noise of what it was.

**That has happened on one dataset and on neither of the two since.** On the
DCIS cohort at 773 genes, ``exp/17``'s run of 2026-11-02 found every arm
ending with the decoder predicting exactly one half for every pair and the
network latent on its prior, so balancing is neither better nor worse than the
plain mean there. On ``dataset_full_1``, ``exp/18`` found every decoder
predicting no edge anywhere under this term, which is the outcome it exists to
prevent. The term is left on because nothing measured is worse under it; read
the claim above as a result on one synthetic network rather than as a property
of the term.

``LossWeights.edge_balance`` chooses how the pairs are scored and
:func:`network_term` computes it. The default is ``"balanced"``; ``"none"`` is
the old term, kept for repeating any graph-model result produced before
2026-10-30. A third way, scoring the linked pairs against as many absent pairs
drawn at random as graph autoencoders do, recovered the network about as well
at more cost and collapsed once where the balanced term never did; it lost,
so by the rule in ``exp/README.md`` it lives in ``exp/17`` and not here.
Decision of 2026-10-30 on weighting the network term, which closed issue #13.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from preffect.config import LossDelays, LossWeights
from preffect.errors import ConfigError
from preffect.model import ModelOutput

__all__ = ["LossTerm", "Loss", "gaussian_kl_divergence", "network_term", "compute_loss"]


@dataclass(frozen=True, slots=True)
class LossTerm:
    """One contribution to the objective, with the weight it carries."""

    name: str
    value: torch.Tensor  # a scalar
    weight: float

    @property
    def weighted(self) -> torch.Tensor:
        return self.weight * self.value


@dataclass(frozen=True, slots=True)
class Loss:
    """The objective and the terms it was built from.

    Keeping the terms means a training loop can report each one without
    recomputing anything, which is how version 0.1's reporting drifted out of
    step with what it was optimising.
    """

    total: torch.Tensor
    terms: tuple[LossTerm, ...]
    normalised: bool

    @property
    def weight_total(self) -> float:
        """The divisor: the weights of the terms actually present, and no others."""
        return sum(term.weight for term in self.terms)

    def named_values(self) -> dict[str, float]:
        """Each term's unweighted value, for logging."""
        return {term.name: float(term.value.detach()) for term in self.terms}

    def named_weighted(self) -> dict[str, float]:
        """Each term's contribution after weighting, for logging."""
        return {term.name: float(term.weighted.detach()) for term in self.terms}


def gaussian_kl_divergence(
    mean: torch.Tensor, log_variance: torch.Tensor
) -> torch.Tensor:
    """Distance from a standard Gaussian, per sample.

    Measures how far the distribution the encoder produced for a sample has
    moved from the standard Gaussian the model assumes before seeing anything.
    Zero when they match.

    :returns: one value per sample, summed over the latent dimensions.
    """
    return -0.5 * torch.sum(
        1 + log_variance - mean.pow(2) - log_variance.exp(), dim=-1
    )


def compute_loss(
    output: ModelOutput,
    counts: torch.Tensor,
    weights: LossWeights,
    delays: LossDelays | None = None,
    epoch: int = 1,
    batch_labels: torch.Tensor | None = None,
    network: torch.Tensor | None = None,
    real_rows: torch.Tensor | None = None,
) -> Loss:
    """Build the objective for one batch.

    :param output: what the model produced.
    :param counts: the counts to score against. During training with masking
        this is the unmasked original, so the model is judged on what it should
        have recovered rather than on what it was shown.
    :param epoch: counted from 1, used against the delays.
    :param batch_labels: required only when the batch centroid term is active.
    :param network: the observed sample-sample network for this batch, as a
        square 0/1 matrix. Supplied for the graph models, absent otherwise, in
        which case the two network terms are not in the objective at all.
    :param real_rows: which rows of ``counts`` are real samples, when the full
        model's data has rows padded in for patients missing a tissue. The
        three terms that read the counts are then averaged over those rows
        only: the expression reconstruction, and the two library-size terms.
        The latent and network terms are averaged over every row, padding
        included. Leave it None, which is what the simple and single models
        always pass, and every term is averaged over every row as before, to
        the last bit. See :mod:`preffect.data.tissues`.
    :raises ConfigError: if ``real_rows`` selects nothing, which means every
        row of this batch is padding and the counts terms have nothing to
        average. Version 0.1 averaged over nothing and produced a loss that
        was not a number; a larger batch size, or dropping the patients who
        are missing a tissue, is what to do instead.
    """
    delays = delays or LossDelays()
    if epoch < 1:
        raise ConfigError(f"epochs count from 1, got {epoch}")
    if real_rows is not None and not bool(real_rows.any()):
        raise ConfigError(
            f"every one of this batch's {real_rows.shape[0]} rows is padding, "
            f"added for a patient missing the predicted tissue, so the "
            f"expression and library terms have nothing to average over. Use a "
            f"larger batch size so that every batch holds a real sample, or "
            f"drop the patients who are missing that tissue."
        )

    terms: list[LossTerm] = []

    def over_real(value: torch.Tensor) -> torch.Tensor:
        """One value per row, averaged over the real rows of this batch."""
        return value.mean() if real_rows is None else value[real_rows].mean()

    def add(name: str, value: torch.Tensor, weight: float) -> None:
        if delays.active_at(name, epoch):
            terms.append(LossTerm(name=name, value=value, weight=weight))

    # The main term: surprise at the counts actually seen, summed across genes
    # so that a sample with more genes expressed is not penalised for it, then
    # averaged across the real samples.
    add(
        "expression_recon",
        over_real(-output.counts.log_prob(counts).sum(dim=-1)),
        weights.expression_recon,
    )

    # Over every row, padding included. Version 0.1 commented out the lines
    # that would have taken the padded rows out of this term and out of the
    # network terms below, and left a note beside the call saying ghost rows
    # "shouldn't be considered in the loss functions (except its ok for the
    # adjacency graphs)". The decision of 2026-11-02 inherits that as it
    # stands and measures it.
    add(
        "expression_kl",
        gaussian_kl_divergence(output.latent_mean, output.latent_log_variance).mean(),
        weights.expression_kl[0],
    )

    # Library terms exist only when the library size is being inferred. When it
    # is read from the data there is nothing to reconstruct and nothing to
    # regularise, so the terms are absent rather than present and zero. That
    # distinction is the point: version 0.1 kept their weights in the divisor
    # while their values were identically zero.
    if output.infers_library_size:
        observed_log = torch.log(counts.sum(dim=1, keepdim=True) + 1)
        predicted_log = output.predicted_log_library_size
        if real_rows is not None:
            observed_log = observed_log[real_rows]
            predicted_log = predicted_log[real_rows]
        add(
            "library_recon",
            torch.nn.functional.mse_loss(predicted_log, observed_log),
            weights.library_recon[0],
        )
        add(
            "library_kl",
            over_real(
                gaussian_kl_divergence(
                    output.library_latent_mean, output.library_latent_log_variance
                )
            ),
            weights.library_kl[0],
        )

    # Present only for the graph models. Absent, not zero, for the rest: a term
    # that does not apply should not be in the divisor either.
    if network is not None and getattr(output, "predicted_network", None) is not None:
        # One term per tissue, each carrying that tissue's own weight. Version
        # 0.1 stacked them and paired the stack against the weight list with
        # zip, which silently dropped a tissue when the list was short; that is
        # finding 4. Here a tissue that is present has a named term and a
        # weight of its own, and the divisor follows from the list.
        networks = network if isinstance(network, (list, tuple)) else [network]
        predicted = output.predicted_network
        predicted = predicted if isinstance(predicted, (list, tuple)) else [predicted]
        net_means = output.network_latent_mean
        net_means = net_means if isinstance(net_means, (list, tuple)) else [net_means]
        net_vars = output.network_latent_log_variance
        net_vars = net_vars if isinstance(net_vars, (list, tuple)) else [net_vars]

        if not (len(networks) == len(predicted) == len(net_means)):
            raise ConfigError(
                f"got {len(networks)} observed network(s) but {len(predicted)} "
                f"predicted and {len(net_means)} network latent space(s)"
            )
        weights.check_tissue_count(len(networks))

        suffix = (lambda i: "") if len(networks) == 1 else (lambda i: f"_{i + 1}")
        for i, (observed, guess) in enumerate(zip(networks, predicted, strict=True)):
            add(
                f"adjacency_recon{suffix(i)}",
                network_term(guess, observed, weights.edge_balance),
                weights.adjacency_recon[i],
            )
            add(
                f"adjacency_kl{suffix(i)}",
                gaussian_kl_divergence(net_means[i], net_vars[i]).mean(),
                weights.adjacency_kl[i],
            )

    if batch_labels is not None:
        add(
            "batch_centroid",
            _batch_centroid_penalty(output.latent, batch_labels),
            weights.batch_centroid[0],
        )

    if not terms:
        raise ConfigError(
            f"no loss term is active at epoch {epoch}, so there is nothing to "
            f"train on. Check the delays: every one of them starts later than "
            f"this epoch."
        )

    weighted_sum = torch.stack([term.weighted for term in terms]).sum()
    divisor = sum(term.weight for term in terms)
    total = weighted_sum / divisor if weights.normalise_by_weight_sum else weighted_sum

    return Loss(
        total=total,
        terms=tuple(terms),
        normalised=weights.normalise_by_weight_sum,
    )


def network_term(
    predicted: torch.Tensor, observed: torch.Tensor, edge_balance: str = "none"
) -> torch.Tensor:
    """How far a predicted sample-sample network is from the observed one, for one batch.

    Binary cross-entropy between the predicted edge probabilities and the
    observed 0/1 network, both square over the samples in the batch. The
    ``edge_balance`` setting of :class:`~preffect.config.LossWeights` says
    which pairs are scored and with what weight, one of two ways:

    - ``"balanced"``, the default: every pair off the diagonal, with each
      linked pair weighted by the number of absent pairs per linked pair in
      this batch, so the linked and the absent pairs carry half the weight
      each. Divided by the total weight, so it is a weighted mean on the same
      scale as ``"none"``: a model that predicts one half everywhere scores
      ln 2 in both.
    - ``"none"``: the plain mean over every entry of the matrix, the diagonal
      included, as version 0.1 scored it. On a network 1 per cent dense that
      is 99 parts "no edge here" to one part "edge here", and under it the
      network half of the model learns nothing (``exp/17``). Kept for
      repeating a result produced before 2026-10-30.

    The balanced mode leaves the diagonal out. A sample's link to itself is
    not a pair, the observed network never carries one (the loader drops
    self-links), and the decoder scores it by the inner product of a vector
    with itself, which is never below one half, so scoring it against zero can
    only push the decoder towards predicting one half everywhere. A batch with
    no linked pair has nothing to balance against and scores its absent pairs
    as a plain mean; a batch of one sample has no pair at all and scores as
    ``"none"`` does.

    :param predicted: edge probabilities, ``[samples, samples]``.
    :param observed: the network for the same samples as a 0/1 matrix.
    :raises ConfigError: for an ``edge_balance`` that is not one of the two.
    """
    guess = predicted.clamp(1e-7, 1 - 1e-7)
    if edge_balance == "none":
        return torch.nn.functional.binary_cross_entropy(guess, observed)
    if edge_balance not in LossWeights.EDGE_BALANCE_MODES:
        raise ConfigError(
            f"edge_balance must be one of {list(LossWeights.EDGE_BALANCE_MODES)}, "
            f"got {edge_balance!r}"
        )
    if guess.shape != observed.shape or guess.dim() != 2 or guess.shape[0] != guess.shape[1]:
        raise ConfigError(
            f"the predicted network is shaped {tuple(guess.shape)} and the observed "
            f"one {tuple(observed.shape)}; both must be square over the same samples"
        )

    per_pair = torch.nn.functional.binary_cross_entropy(guess, observed, reduction="none")
    n = observed.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=observed.device)
    present = (observed > 0) & off_diagonal
    absent = (observed <= 0) & off_diagonal
    n_present = int(present.sum())
    n_absent = int(absent.sum())
    if n_present == 0 or n_absent == 0:
        pairs = off_diagonal if n > 1 else torch.ones_like(off_diagonal)
        return per_pair[pairs].mean()

    weight = absent.float() + (n_absent / n_present) * present.float()
    return (per_pair * weight).sum() / weight.sum()


def _batch_centroid_penalty(
    latent: torch.Tensor, batch_labels: torch.Tensor
) -> torch.Tensor:
    """Pull samples from the same batch towards their shared centre.

    Mean squared distance from each sample's latent point to the average point
    of its batch. Lowering it makes batches more compact, which is one way of
    asking the model to describe biology rather than sequencing run.
    """
    labels = batch_labels.reshape(-1).long()
    if labels.shape[0] != latent.shape[0]:
        raise ConfigError(
            f"got {labels.shape[0]} batch labels for {latent.shape[0]} samples"
        )

    total = latent.new_zeros(())
    for label in torch.unique(labels):
        members = latent[labels == label]
        total = total + ((members - members.mean(dim=0)) ** 2).sum(dim=-1).mean()
    return total / len(torch.unique(labels))
