"""The simple model: counts in, a fitted count distribution out.

The simple model uses the counts alone, with optional correction variables such
as sequencing batch. The single and full models add a sample-sample network and
matched tissues, and arrive later.

## What it does, in order

1. The **encoder** turns a sample's counts into a mean and a variance over a
   small latent space, a handful of numbers summarising that sample.
2. **Reparameterising** draws a point from that distribution in a way that
   gradients can pass through, so the model can be trained.
3. The **decoder** turns the drawn point back into a hidden representation.
4. Three **heads** turn that representation into the parameters of a count
   distribution per gene: the composition, the dispersion, and, for the
   zero-inflated form, the extra-zero log-odds.
5. Separately, a **library size** is either taken from the data or predicted by
   its own small encoder and decoder. Multiplying the composition by it gives
   the expected count.

The composition comes out of a softmax, so a sample's predicted proportions sum
to one across genes and the library size sets the scale. That split is what
lets the model describe what a sample is made of separately from how deeply it
was sequenced.

Because the library sets the scale and the composition cannot change it, a
library read off a matrix with entries hidden makes every predicted count
short by the hidden fraction, and no amount of training can make it up.
``forward`` therefore takes a ``library_scale``: the factor that estimates a
sample's real total back from how many of its entries were hidden. It is None
when nothing is hidden, and the model is then exactly what it was. Finding 56.

## Output

:class:`ModelOutput` holds every intermediate quantity as a named field.
Version 0.1 returned nested dictionaries of lists keyed by strings, so a
consumer reached for ``outputs['latent_spaces']['mu_simple'][0]`` and a typo
produced a ``KeyError`` at best.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from preffect.config import Likelihood, ModelConfig
from preffect.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from preffect.layers import (
    CountDecoder,
    CountEncoder,
    LibrarySizeDecoder,
    LibrarySizeEncoder,
)

__all__ = ["ModelOutput", "SimpleModel"]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Everything one forward pass produced.

    Kept whole rather than reduced, because the loss needs the latent
    distribution, inference needs the fitted counts, and diagnostics need both.
    """

    #: Latent summary of each sample: mean, log variance, and the drawn point.
    latent_mean: torch.Tensor
    latent_log_variance: torch.Tensor
    latent: torch.Tensor

    #: Predicted proportion of each sample's counts belonging to each gene.
    #: Sums to one across genes.
    composition: torch.Tensor
    #: Inverse dispersion per gene per sample, positive.
    dispersion: torch.Tensor
    #: Total counts per sample used to scale the composition.
    library_size: torch.Tensor
    #: The fitted distribution over counts.
    counts: NegativeBinomial | ZeroInflatedNegativeBinomial

    #: Log-odds of an extra zero. Present only for the zero-inflated likelihood.
    zero_logits: torch.Tensor | None = None

    #: The library size latent space, present only when it is being inferred.
    library_latent_mean: torch.Tensor | None = None
    library_latent_log_variance: torch.Tensor | None = None
    library_latent: torch.Tensor | None = None
    predicted_log_library_size: torch.Tensor | None = None

    @property
    def infers_library_size(self) -> bool:
        return self.library_latent_mean is not None

    @property
    def expected_counts(self) -> torch.Tensor:
        """The mean of the fitted distribution.

        Returns the product the distribution already holds rather than
        recomputing it. ``_fitted_counts`` multiplies composition by library
        size to build the distribution, and this property used to do the same
        multiplication again on every access, over a samples-by-genes tensor.
        Reading it three times meant three identical multiplications of four
        megabytes.

        It also means the two can no longer disagree. They were computed
        separately from the same inputs, so nothing forced them to match.
        """
        return self.counts.mu


class SimpleModel(nn.Module):
    """The simple PREFFECT model.

    :param n_genes: number of genes, fixed once the data is read.
    :param config: the model section of a checked configuration.
    """

    def __init__(self, n_genes: int, config: ModelConfig):
        super().__init__()
        self.n_genes = n_genes
        self.config = config

        n_corrections = len(config.correction_variables)

        self.encoder = CountEncoder(
            n_genes=n_genes,
            n_corrections=n_corrections,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            leaky_relu_slope=config.leaky_relu_slope,
            dropout=config.dropout,
        )
        self.decoder = CountDecoder(
            latent_dim=config.latent_dim,
            n_corrections=n_corrections,
            hidden_dim=config.hidden_dim,
            leaky_relu_slope=config.leaky_relu_slope,
            dropout=config.dropout,
        )

        if config.infer_library_size:
            self.library_encoder = LibrarySizeEncoder(
                n_corrections=n_corrections,
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
                leaky_relu_slope=config.leaky_relu_slope,
                dropout=config.dropout,
            )
            self.library_decoder = LibrarySizeDecoder(
                latent_dim=config.latent_dim,
                n_corrections=n_corrections,
                hidden_dim=config.hidden_dim,
                leaky_relu_slope=config.leaky_relu_slope,
            )
        else:
            self.library_encoder = None
            self.library_decoder = None

        self.to_composition = nn.Linear(config.hidden_dim, n_genes)
        self.to_dispersion = nn.Linear(config.hidden_dim, n_genes)
        self.to_zero_logits = (
            nn.Linear(config.hidden_dim, n_genes)
            if config.likelihood is Likelihood.ZINB
            else None
        )

        self.activation = nn.LeakyReLU(negative_slope=config.leaky_relu_slope)

    # -- seams -------------------------------------------------------------

    def activate_decoded(self, hidden: torch.Tensor) -> torch.Tensor:
        """Shape the decoder's output before the parameter heads read it.

        Nothing, deliberately. The decoder already ends with a leaky rectifier,
        and version 0.1 applied a second one here that nothing in the code or
        the paper explains. It is removed because one unexplained non-linearity
        fewer is simpler and the reconstruction is unaffected.

        **That is a judgement about simplicity, not a measured improvement,**
        and two earlier versions of this docstring claimed more. ``exp/03`` does
        find the same reconstruction for less than half the KL divergence
        (305.7 against 143.4 across eight seeds), but the KL divergence is not a
        property of the model alone: rescaling the latent and compensating in
        the decoder's first layer leaves the model's output identical and moves
        the KL by tens of per cent. See finding 30. A claim that the lower-KL
        latent space clusters better was also withdrawn, untested.

        Still a seam, so the comparison can be repeated at full scale, and the
        repeat should measure something functional rather than the KL. See
        ``decisions/2026-10-15_exp03_second_rectifier_removed.md``.
        """
        return hidden

    # -- the pass ----------------------------------------------------------

    def reparameterise(
        self, mean: torch.Tensor, log_variance: torch.Tensor
    ) -> torch.Tensor:
        """Draw from a Gaussian so that gradients still reach the parameters.

        Sampling directly would break the chain, because a draw is not a
        differentiable function of the distribution it came from. Drawing
        standard noise and scaling it keeps the randomness outside the path the
        gradient takes.
        """
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def forward(
        self,
        counts: torch.Tensor,
        corrections: torch.Tensor | None = None,
        library_scale: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Run one batch through the model.

        :param counts: ``[samples, genes]``, raw counts.
        :param corrections: ``[samples, correction variables]``, or None.
        :param library_scale: ``[samples, 1]``, what each sample's observed
            total is multiplied by to estimate the total it would have had
            with nothing hidden. Supplied when entries have been masked, and
            None otherwise, in which case nothing changes. Finding 56.
        """
        if counts.shape[1] != self.n_genes:
            raise ValueError(
                f"this model was built for {self.n_genes} genes but was given "
                f"{counts.shape[1]}"
            )

        cfg = self.config

        latent_mean, latent_log_variance = self.encoder(counts, corrections)
        latent_mean = latent_mean.clamp(-cfg.max_latent_mean, cfg.max_latent_mean)
        latent_log_variance = latent_log_variance.clamp(
            -cfg.max_log_variance, cfg.max_log_variance
        )
        latent = self.reparameterise(latent_mean, latent_log_variance)

        return self._finish_counts(
            counts,
            corrections,
            latent,
            latent_mean,
            latent_log_variance,
            library_scale=library_scale,
        )

    def _finish_counts(
        self,
        counts: torch.Tensor,
        corrections: torch.Tensor | None,
        latent: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        library_scale: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Everything downstream of the latent values.

        Shared by all three models, and called again at inference time with the
        latent mean in place of a draw, so that a cleaned count matrix is
        reproducible.
        """
        library = self._library_size(counts, corrections, library_scale)
        hidden = self.activate_decoded(self.decoder(latent, corrections))

        composition = torch.softmax(self.to_composition(hidden), dim=1)
        dispersion = self._dispersion(hidden)
        zero_logits = (
            self.to_zero_logits(hidden) if self.to_zero_logits is not None else None
        )

        return ModelOutput(
            latent_mean=mean,
            latent_log_variance=log_variance,
            latent=latent,
            composition=composition,
            dispersion=dispersion,
            library_size=library["library_size"],
            counts=self._fitted_counts(composition, dispersion, zero_logits, library),
            zero_logits=zero_logits,
            library_latent_mean=library["latent_mean"],
            library_latent_log_variance=library["latent_log_variance"],
            library_latent=library["latent"],
            predicted_log_library_size=library["predicted_log"],
        )

    # -- pieces ------------------------------------------------------------

    def _fitted_counts(
        self,
        composition: torch.Tensor,
        dispersion: torch.Tensor,
        zero_logits: torch.Tensor | None,
        library: dict[str, torch.Tensor | None],
    ):
        """The count distribution the composition and library size imply."""
        mean_counts = composition * library["library_size"]
        if self.config.likelihood is Likelihood.ZINB:
            return ZeroInflatedNegativeBinomial(
                mu=mean_counts, theta=dispersion, zero_logits=zero_logits
            )
        return NegativeBinomial(mu=mean_counts, theta=dispersion)

    def _dispersion(self, hidden: torch.Tensor) -> torch.Tensor:
        """Positive inverse dispersion per gene per sample."""
        raw = self.to_dispersion(hidden)
        if not self.config.log_scale_dispersion:
            return torch.nn.functional.softplus(raw)
        limit = self.config.max_log_library_size
        return torch.exp(raw.clamp(-limit, limit))

    def _library_size(
        self,
        counts: torch.Tensor,
        corrections: torch.Tensor | None,
        library_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Either the observed total per sample, or one the model predicts.

        The counts this reads are what the model was shown, which under a mask
        is short of the sample's real total by whatever the hiding took away.
        ``library_scale`` is the factor that estimates the real total back,
        from the **number** of entries hidden in each sample and never from
        their values. Without it the composition, which sums to one across
        genes, is multiplied by a total a fifth too small at a fifth hidden,
        and nothing in the model can recover the difference. Finding 56.

        The correction applies to both paths, because both start from the
        observed total: directly when the library is read from the data, and
        as the log total handed to the library encoder when it is inferred.
        """
        observed = counts.sum(dim=1, keepdim=True)
        if library_scale is not None:
            observed = observed * library_scale

        if self.library_encoder is None:
            return {
                "library_size": observed,
                "latent_mean": None,
                "latent_log_variance": None,
                "latent": None,
                "predicted_log": None,
            }

        cfg = self.config
        log_observed = torch.log(observed.squeeze(1) + 1)
        mean, log_variance = self.library_encoder(log_observed, corrections)
        log_variance = log_variance.clamp(-cfg.max_log_variance, cfg.max_log_variance)
        latent = self.reparameterise(mean, log_variance)

        predicted_log = self.library_decoder(latent, corrections)
        limit = cfg.max_log_library_size
        return {
            "library_size": torch.exp(predicted_log.clamp(-limit, limit)),
            "latent_mean": mean,
            "latent_log_variance": log_variance,
            "latent": latent,
            "predicted_log": predicted_log,
        }
