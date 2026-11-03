"""Applying a fitted model: cleaned counts, latent summaries, imputation.

Training fits the model. Inference uses it: run data through and keep what
comes out, as cleaned counts to hand to an ordinary analysis, as a latent
summary to cluster in, or as filled-in values where counts were missing.

## Truth is passed in, never looked up

:func:`imputation_error` takes the true values as an argument. It cannot find
them for you, and that is the whole point.

Version 0.1 looked for a layer called ``truth`` in the input, and if it was not
there the data loader quietly made one up: the per-gene mean across samples.
The error routine then compared the model against that, in a column it labelled
``truth_mu``. On any real dataset, which carries no simulation parameters, this
reported distance from the gene mean under the name of imputation accuracy, and
nothing warned. That is finding 10.

Here there is nowhere for a fabricated truth to come from. If you have no true
values you cannot call the function, which is the correct outcome.

## Rows that stand in for a missing tissue are not reported

The full model can be given a cohort where some patients lack the tissue it
predicts. A row of per-gene means stands in for each of those patients so the
rest of the cohort can be encoded, and :func:`infer` drops those rows from
everything it returns and says how many it dropped. Reporting a cleaned count
for a profile nobody measured is the sort of thing finding 10 was about.

## Inference uses the latent mean, not a draw

Version 0.1 ran inference through the same path as training, which **samples**
from the latent distribution, so running the same model over the same data
twice gave different cleaned counts. Here the latent mean is used by default,
so the output is deterministic and a result can be reproduced. Pass
``sample_latent=True`` for version 0.1's behaviour, which is what you want if
you are deliberately drawing several times to see the spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from preffect.errors import DataError, PreffectError

__all__ = [
    "InferenceResult",
    "NetworkRecovery",
    "infer",
    "imputation_error",
    "correlation_per_gene",
    "log_likelihood_per_sample",
    "network_recovery",
]


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """What a fitted model made of some data."""

    #: Expected counts: the model's cleaned view of the input.
    expected_counts: torch.Tensor
    #: Predicted proportion of each sample's counts per gene, summing to one.
    composition: torch.Tensor
    #: Inverse dispersion per gene per sample.
    dispersion: torch.Tensor
    #: Total counts per sample used to scale the composition.
    library_size: torch.Tensor
    #: Latent summary per sample, the thing worth clustering in.
    latent: torch.Tensor
    #: Log-odds of an extra zero, for the zero-inflated likelihood only.
    zero_logits: torch.Tensor | None = None
    #: True when the latent mean was used rather than a draw from it.
    deterministic: bool = True
    #: How many rows were left out because they were padding rather than
    #: samples. The full model can be given a cohort where some patients lack
    #: the predicted tissue; a row of per-gene means stands in for each while
    #: the model runs, and nothing about it is reported here. Zero for every
    #: other case. See :mod:`preffect.data.tissues`.
    n_padded_rows_dropped: int = 0

    @property
    def n_samples(self) -> int:
        return self.expected_counts.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expected_counts.shape[1]

    def to_anndata(
        self,
        sample_names: list[str] | None = None,
        gene_names: list[str] | None = None,
    ):
        """Package the result as an AnnData, for handing to other tools.

        ``X`` holds the expected counts. The rest travel as layers and as
        ``obsm`` so nothing has to be recomputed.
        """
        import anndata as ad
        import pandas as pd

        def frame(names, n, prefix):
            index = names if names is not None else [f"{prefix}{i}" for i in range(n)]
            if len(index) != n:
                raise DataError(
                    f"got {len(index)} {prefix}names for {n} {prefix}s"
                )
            # Object dtype on purpose. Under pandas 3 a list of names becomes a
            # string array, which anndata 0.11 refuses to write without an
            # opt-in, so `preffect apply` crashed at its last line in a fresh
            # install while passing in this repository's newer environment.
            # Finding 48.
            return pd.DataFrame(index=pd.Index(list(index), dtype=object))

        adata = ad.AnnData(
            X=self.expected_counts.detach().cpu().numpy().astype(np.float32),
            obs=frame(sample_names, self.n_samples, "sample_"),
            var=frame(gene_names, self.n_genes, "gene_"),
        )
        adata.layers["composition"] = self.composition.detach().cpu().numpy()
        adata.layers["dispersion"] = self.dispersion.detach().cpu().numpy()
        if self.zero_logits is not None:
            adata.layers["zero_logits"] = self.zero_logits.detach().cpu().numpy()
        adata.obsm["latent"] = self.latent.detach().cpu().numpy()
        adata.obs["library_size"] = self.library_size.detach().cpu().numpy().ravel()
        adata.uns["deterministic"] = self.deterministic
        adata.uns["padded_rows_dropped"] = self.n_padded_rows_dropped
        return adata


@torch.no_grad()
def infer(
    model: nn.Module,
    data,
    sample_latent: bool = False,
) -> InferenceResult:
    """Run a fitted model over some data.

    Runs the whole split in one pass rather than in mini-batches: there is no
    optimiser step, so there is nothing to gain by dividing it, and batching
    would cut the sample-sample network down for no reason. See finding 22.

    :param data: a :class:`~preffect.data.tensors.TensorData` for the simple
        and single models, or a
        :class:`~preffect.data.tissues.MultiTissueData` for the full model.
    :param sample_latent: draw from the latent distribution instead of using
        its mean. Off by default, so the same model over the same data gives
        the same answer twice.
    :returns: one row per **real** sample of the predicted tissue. Where the
        full model was given a cohort with patients missing that tissue, a row
        of per-gene means stood in for each of them so that the rest of the
        cohort could be encoded; those rows are dropped here and counted in
        ``n_padded_rows_dropped``, because nothing in them was measured.
    """
    from preffect.data.tissues import MultiTissueData

    # Where the model is, the data goes. ``train`` moves its own copy of the
    # data to the device and the caller's copy stays where it was, so the
    # natural sequence of fitting on a GPU and then calling this with the same
    # data failed on a device mismatch with a PyTorch error, not a sentence.
    # Finding 45's third half.
    data = data.to(next(model.parameters()).device)

    was_training = model.training
    model.eval()
    try:
        # The library-size correction travels with the data, as the factor that
        # estimates back what hiding entries took off each sample's total. It
        # is None when nothing was hidden. For the full model it is the target
        # tissue's, since that is the only tissue whose counts are predicted.
        # Finding 56.
        real_rows = None
        if isinstance(data, MultiTissueData):
            counts = [t.inputs for t in data.tissues]
            corrections = data.target.corrections
            library_scale = data.target.library_scale
            output = model(
                counts,
                corrections,
                [t.edge_index for t in data.tissues],
                library_scale=library_scale,
            )
            target_counts = counts[0]
            # Every row goes through the model, padding included, because a
            # padded row is a node of the network the real rows attend over.
            # Only the results are cut down, below.
            real_rows = data.target_real_rows()
        else:
            corrections = data.corrections
            target_counts = data.inputs
            library_scale = data.library_scale
            if data.has_network:
                output = model(
                    data.inputs, corrections, data.edge_index, library_scale=library_scale
                )
            else:
                output = model(data.inputs, corrections, library_scale=library_scale)

        if sample_latent:
            latent = output.latent
        else:
            # Decode again from the mean, so every reported quantity comes from
            # the same latent values rather than the composition coming from a
            # draw and the summary from the mean.
            latent = output.latent_mean
            output = model._finish_counts(  # noqa: SLF001 - the model's own helper
                target_counts,
                corrections,
                latent,
                output.latent_mean,
                output.latent_log_variance,
                library_scale=library_scale,
            )
    finally:
        model.train(was_training)

    def keep(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None or real_rows is None:
            return value
        return value[real_rows]

    return InferenceResult(
        expected_counts=keep(output.expected_counts),
        composition=keep(output.composition),
        dispersion=keep(output.dispersion),
        library_size=keep(output.library_size),
        latent=keep(latent),
        zero_logits=keep(output.zero_logits),
        deterministic=not sample_latent,
        n_padded_rows_dropped=0 if real_rows is None else int((~real_rows).sum()),
    )


def log_likelihood_per_sample(result: InferenceResult, observed) -> np.ndarray:
    """How well the fitted distribution explains each sample, one number per sample.

    The mean log probability **per gene** of that sample's observed counts
    under the distribution the model fitted for it. Per gene rather than
    summed over genes, so the number does not grow with the size of the panel
    and two cohorts can be compared.

    Larger is better, and every value is negative, because these are
    probabilities of a particular count. The number on its own means little;
    what it is for is ranking the cohort. A sample far below the rest is one
    the model cannot account for, and the three usual reasons are a failed
    library, a mislabelled sample, and a sample of a different tissue.

    The distribution is rebuilt from what :func:`infer` returned, so it is the
    same negative binomial, or zero-inflated negative binomial, that the
    objective was computed with.

    :param observed: the counts the model was given, samples by genes.
    :raises PreffectError: if the counts are not the shape the result is.
    """
    from preffect.distributions import (
        NegativeBinomial,
        ZeroInflatedNegativeBinomial,
    )

    counts = _as_tensor(observed)
    mu = result.expected_counts.detach().cpu().float()
    if counts.shape != mu.shape:
        raise PreffectError(
            f"the counts are shaped {tuple(counts.shape)} and the fitted means "
            f"{tuple(mu.shape)}; they have to match"
        )
    theta = result.dispersion.detach().cpu().float()
    if result.zero_logits is None:
        fitted = NegativeBinomial(mu=mu, theta=theta)
    else:
        fitted = ZeroInflatedNegativeBinomial(
            mu=mu, theta=theta, zero_logits=result.zero_logits.detach().cpu().float()
        )
    return fitted.log_prob(counts).mean(dim=1).detach().cpu().numpy()


def correlation_per_gene(result: InferenceResult, observed) -> np.ndarray:
    """How closely fitted follows observed across the samples, one number per gene.

    Pearson correlation down each column: for one gene, the observed count in
    every sample against the count the model fitted for that same sample. One
    means the model tracks that gene's variation across the cohort perfectly,
    and zero means it predicts the same thing for every sample.

    A gene that is constant in the counts, or constant in the fit, has no
    correlation to report and comes back as ``nan`` rather than as a zero that
    would read as a failure. Take the median over the genes that are not
    ``nan`` for a single number describing a fit.

    :param observed: the counts the model was given, samples by genes.
    :raises PreffectError: if the counts are not the shape the result is.
    """
    counts = _as_tensor(observed).numpy()
    fitted = result.expected_counts.detach().cpu().float().numpy()
    if counts.shape != fitted.shape:
        raise PreffectError(
            f"the counts are shaped {counts.shape} and the fitted counts "
            f"{fitted.shape}; they have to match"
        )
    centred_observed = counts - counts.mean(axis=0)
    centred_fitted = fitted - fitted.mean(axis=0)
    # Spread of zero means the correlation is undefined, not zero. Dividing
    # anyway would turn "this gene never moves" into "the model got it wrong".
    spread = np.sqrt((centred_observed**2).sum(axis=0) * (centred_fitted**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = (centred_observed * centred_fitted).sum(axis=0) / spread
    correlation[spread == 0] = np.nan
    return correlation


def _as_tensor(values) -> torch.Tensor:
    """Counts as a float tensor, whatever the caller is holding them as."""
    if hasattr(values, "detach"):
        return values.detach().cpu().float()
    if hasattr(values, "toarray"):
        values = values.toarray()
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def imputation_error(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    hidden: torch.Tensor | None = None,
    kind: str = "mean_squared",
) -> dict[str, float]:
    """Compare filled-in values against values that are actually known.

    :param predicted: the model's expected counts.
    :param truth: the real values. **Required, and never inferred.** If you do
        not have them, you cannot measure imputation accuracy, and version
        0.1's habit of substituting the per-gene mean is why this is spelled
        out. See finding 10.
    :param hidden: boolean mask of the entries that were hidden from the model.
        When given, the error is reported separately for hidden and visible
        entries, which is the comparison that means anything: doing well on
        entries the model was shown is not imputation.
    :param kind: ``mean_squared`` or ``mean_relative``.
    :raises PreffectError: on a shape mismatch or an unknown kind.
    """
    if predicted.shape != truth.shape:
        raise PreffectError(
            f"predicted is shaped {tuple(predicted.shape)} and truth "
            f"{tuple(truth.shape)}; they have to match"
        )
    if kind not in ("mean_squared", "mean_relative"):
        raise PreffectError(
            f"kind must be 'mean_squared' or 'mean_relative', got {kind!r}"
        )

    predicted = predicted.detach().float()
    truth = truth.detach().float()

    def score(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() == 0:
            return float("nan")
        if kind == "mean_squared":
            return float(((a - b) ** 2).mean())
        # Relative error is undefined where the true value is zero, and count
        # data is 10 to 55 per cent zeros. Version 0.1 divided by
        # ``truth + 1e-8``, which makes a zero-truth entry contribute up to a
        # hundred million. It rarely fired there, because version 0.1 compared
        # against smoothed per-gene means rather than counts, and a per-gene
        # mean is zero only for a gene that is zero in every sample: one such
        # gene exists across the nine datasets. Comparing against actual
        # counts, which is the correct thing, makes it ordinary. Those entries
        # are left out and counted. See finding 25.
        nonzero = b != 0
        if not nonzero.any():
            return float("nan")
        return float((a[nonzero] - b[nonzero]).abs().div(b[nonzero].abs()).mean())

    if hidden is not None and hidden.shape != truth.shape:
        raise PreffectError(
            f"the mask is shaped {tuple(hidden.shape)} and truth "
            f"{tuple(truth.shape)}; they have to match"
        )
    result = {"overall": score(predicted, truth)}
    if kind == "mean_relative":
        # Counted over the entries that were scored: the hidden ones when a
        # mask is given, the whole matrix otherwise. It used to be counted over
        # the whole matrix whatever was scored (finding 53).
        scored = truth[hidden] if hidden is not None else truth
        skipped = int((scored == 0).sum())
        result["zero_truth_entries_skipped"] = skipped
        result["fraction_skipped"] = skipped / scored.numel() if scored.numel() else 0.0

    if hidden is not None:
        result["hidden"] = score(predicted[hidden], truth[hidden])
        result["visible"] = score(predicted[~hidden], truth[~hidden])
        result["n_hidden"] = int(hidden.sum())

    return result


@dataclass(frozen=True, slots=True)
class NetworkRecovery:
    """How well a predicted sample-sample network matches the observed one.

    Every number is over the pairs of distinct samples, each pair counted once.
    Two baselines are worth holding beside the scores. A model that predicts
    no edge anywhere predicts nothing, so its precision, recall and F1 are all
    zero. A model that cannot rank the pairs at all, whether it predicts no
    edge, every edge, or one half everywhere, has an average precision equal to
    the density, which is why the density is carried here under both names.
    """

    #: Of the pairs predicted to be edges, the share that are. Zero when nothing
    #: is predicted.
    precision: float
    #: Of the observed edges, the share predicted.
    recall: float
    #: The harmonic mean of the two; zero when either is.
    f1: float
    #: Precision averaged over the ranking of every pair by its probability,
    #: so it needs no threshold. One when every edge outranks every non-edge.
    average_precision: float
    #: The share of pairs that are edges in the observed network.
    density: float
    #: What a prediction that ranks nothing scores on average precision: the
    #: density. The number ``average_precision`` has to beat.
    flat_average_precision: float
    #: The pairs scored, the observed edges among them, and the pairs predicted
    #: to be edges at the threshold of one half.
    n_pairs: int
    n_edges: int
    n_predicted: int

    def __str__(self) -> str:
        return (
            f"precision {self.precision:.3f}, recall {self.recall:.3f}, F1 {self.f1:.3f} "
            f"at one half ({self.n_predicted} pairs predicted, {self.n_edges} edges "
            f"among {self.n_pairs} pairs); average precision {self.average_precision:.3f} "
            f"against {self.flat_average_precision:.3f} for a flat prediction"
        )


def network_recovery(predicted: torch.Tensor, observed: torch.Tensor) -> NetworkRecovery:
    """Score a predicted sample-sample network against the observed one.

    The loss cannot say whether the network is being recovered: on a network
    one per cent dense, a model that predicts no edge anywhere scores well on
    an unweighted cross-entropy (issue #13). This scores the edges directly.

    Both inputs are square over the same samples, the prediction a matrix of
    edge probabilities and the observation a 0/1 matrix. Both are read as
    symmetric: each pair of distinct samples is taken once, from above the
    diagonal, and the diagonal is ignored, since a sample's link to itself is
    not a pair and the loader never carries one. An edge is predicted where the
    probability exceeds one half; at exactly one half the model is undecided
    and no edge is counted, which is what the inner-product decoder produces
    when its latent has collapsed.

    Average precision is computed here rather than imported: sort the pairs by
    probability, and average the precision at each distinct probability over
    the recall it adds, which is the usual definition and treats tied
    probabilities as one threshold.

    :raises PreffectError: if the shapes differ, are not square, or the
        observed network is not symmetric.
    """
    predicted = predicted.detach().float().cpu()
    observed = observed.detach().float().cpu()
    if predicted.shape != observed.shape or predicted.dim() != 2:
        raise PreffectError(
            f"predicted is shaped {tuple(predicted.shape)} and observed "
            f"{tuple(observed.shape)}; both must be square over the same samples"
        )
    n = observed.shape[0]
    if observed.shape != (n, n):
        raise PreffectError(f"the network is shaped {tuple(observed.shape)}, not square")
    if not torch.equal(observed > 0, observed.t() > 0):
        raise PreffectError(
            "the observed network is not symmetric; the models score undirected "
            "networks, with both directions of every edge present"
        )

    rows, cols = torch.triu_indices(n, n, offset=1)
    scores = predicted[rows, cols]
    truth = observed[rows, cols] > 0
    n_pairs = int(truth.numel())
    n_edges = int(truth.sum())
    density = n_edges / n_pairs if n_pairs else 0.0

    called = scores > 0.5
    n_predicted = int(called.sum())
    hits = int((called & truth).sum())
    precision = hits / n_predicted if n_predicted else 0.0
    recall = hits / n_edges if n_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if n_edges and n_pairs:
        order = torch.argsort(scores, descending=True)
        ranked_scores = scores[order]
        ranked_truth = truth[order].float()
        true_so_far = torch.cumsum(ranked_truth, dim=0)
        seen_so_far = torch.arange(1, n_pairs + 1, dtype=torch.float32)
        # The last position of each run of equal scores is one threshold; the
        # precision and recall there are what a cut at that score would give.
        last_of_run = torch.ones(n_pairs, dtype=torch.bool)
        last_of_run[:-1] = ranked_scores[:-1] != ranked_scores[1:]
        precision_at = (true_so_far / seen_so_far)[last_of_run]
        recall_at = (true_so_far / n_edges)[last_of_run]
        recall_before = torch.cat([recall_at.new_zeros(1), recall_at[:-1]])
        average_precision = float(((recall_at - recall_before) * precision_at).sum())
    else:
        average_precision = 0.0

    return NetworkRecovery(
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=average_precision,
        density=density,
        flat_average_precision=density,
        n_pairs=n_pairs,
        n_edges=n_edges,
        n_predicted=n_predicted,
    )
