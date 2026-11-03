"""The network pieces the simple model is built from.

Four of them: an encoder and decoder for the counts, and a smaller pair for the
library size. The graph versions, which replace the count encoder for the
single and full models, arrive with those models.

**Each class builds only what it uses.** Version 0.1 had one encoder class that
constructed the graph path and the plain path together and then chose between
them at call time. For a simple model on twenty thousand genes that allocated a
graph attention layer of sixteen million parameters which was never touched:
measured at 94% of the encoder's weights, 64 MB, initialised on every
construction and written into every saved session. Separate classes cost a
little duplication and remove all of that.

## Terms used here

- **Encoder** maps counts to a small set of numbers, the *latent space*, that
  summarise a sample. **Decoder** maps those numbers back out.
- **Latent space** has a mean and a variance per dimension, because the model
  is variational: it learns a distribution over the summary rather than a
  single value.
- **Correction variables** are things like sequencing batch that the model is
  told about so it can account for them.
- **FiLM**, feature-wise linear modulation, lets a correction variable scale
  and shift every feature rather than merely being pasted on as extra columns.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "scale_counts",
    "FiLM",
    "CountEncoder",
    "CountDecoder",
    "LibrarySizeEncoder",
    "LibrarySizeDecoder",
]


class FiLM(nn.Module):
    """Scale and shift every feature, by an amount the conditioning decides.

    Produces one multiplier and one offset per feature from the conditioning
    values, so a correction variable can modulate the whole representation
    instead of being concatenated to it.
    """

    def __init__(self, n_conditions: int, n_features: int):
        super().__init__()
        self.to_scale_and_shift = nn.Linear(n_conditions, 2 * n_features)

    def forward(self, features: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_and_shift(conditions.float()).chunk(2, dim=-1)
        return scale * features + shift


def _orthogonal_init(module: nn.Module) -> None:
    """Start linear layers with orthogonal weights and a small positive bias.

    Orthogonal initialisation keeps the scale of a signal roughly constant as
    it passes through a layer, which matters here because the counts span
    several orders of magnitude. The same choice as version 0.1.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


def scale_counts(counts: torch.Tensor) -> torch.Tensor:
    """What the expression encoder is fed: ``log(1 + counts)``.

    Version 0.1 fed raw counts, which reach 2.4 million in the DCIS cohort. A
    learning-rate-sized change to one first-layer weight times an input that
    size moved the encoder's output a thousandfold in a single step, which was
    the loss excursion of issue #22 (finding 49). On the log scale the same
    step moves it by a few thousandths, and the spikes are gone in six seeds
    of six (``exp/10``, decision 2026-10-28), with imputation error halved and
    clustering recovery doubled. The targets, the decoder and the objective
    still see raw counts; only the encoder's first layer sees this.

    A function, not a setting, so there is one behaviour. ``exp/10``'s raw arm
    overrides the encoders' ``scale_input`` to feed counts unchanged.
    """
    return torch.log1p(counts)


class CountEncoder(nn.Module):
    """Counts to a latent mean and log variance, for the simple model.

    Correction variables are appended to the input and also drive two FiLM
    layers. Appending them and modulating with them is version 0.1's
    arrangement and is kept.
    """

    def __init__(
        self,
        n_genes: int,
        n_corrections: int,
        hidden_dim: int,
        latent_dim: int,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_corrections = n_corrections
        input_dim = n_genes + n_corrections
        wide = hidden_dim + n_corrections

        self.layer1 = nn.Linear(input_dim, wide)
        self.layer2 = nn.Linear(wide, hidden_dim)
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_log_variance = nn.Linear(hidden_dim, latent_dim)

        self.film1 = FiLM(n_corrections, wide) if n_corrections else None
        self.film2 = FiLM(n_corrections, hidden_dim) if n_corrections else None

        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.apply(_orthogonal_init)

    def scale_input(self, counts: torch.Tensor) -> torch.Tensor:
        """The scale the first layer sees. See :func:`scale_counts`; a seam for ``exp/10``."""
        return scale_counts(counts)

    def combine_with_corrections(
        self, counts: torch.Tensor, corrections: torch.Tensor | None
    ) -> torch.Tensor:
        """Put the correction variables together with the counts.

        A seam, not a setting. Version 0.1 concatenates their raw values here
        while embedding the categorical ones in the library encoder, with no
        explanation for the difference. Experiment ``exp/02`` overrides this to
        try the embedding treatment in both. See
        ``decisions/2026-10-15_odd_choices_kept_pending_measurement.md``.
        """
        if not self.n_corrections:
            return counts
        if corrections is None:
            raise ValueError(
                f"this encoder was built for {self.n_corrections} correction "
                f"variable(s) but none were supplied"
            )
        return torch.cat((counts, corrections.float()), dim=1)

    def project_to_latent(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hidden representation to a latent mean and log variance.

        No activation, unlike version 0.1, which applied a leaky rectifier to
        both. ``exp/01`` measured that across eight seeds and found no
        detectable difference in the loss, the reconstruction or the KL
        divergence term, so the simpler and more conventional form was kept.
        See ``decisions/2026-10-15_exp01_latent_rectifier_removed.md``, which
        also records the limit of that evidence: it was measured at 150 genes
        and wants repeating at realistic scale.

        Still a seam, so that repeat costs an hour rather than a day.
        """
        return self.to_mean(hidden), self.to_log_variance(hidden)

    def forward(
        self, counts: torch.Tensor, corrections: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:returns: the latent mean and log variance, each ``[samples, latent_dim]``."""
        h = self.combine_with_corrections(self.scale_input(counts), corrections)

        h = self.layer1(h)
        h = torch.nn.functional.elu(h)
        if self.film1 is not None:
            h = self.film1(h, corrections)
        h = self.dropout1(h)

        h = self.layer2(h)
        h = torch.nn.functional.elu(h)
        if self.film2 is not None:
            h = self.film2(h, corrections)
        h = self.dropout2(h)

        return self.project_to_latent(h)


class CountDecoder(nn.Module):
    """Latent values back to a hidden representation, for the simple model.

    Stops at the hidden layer. The model turns that into distribution
    parameters, because the three parameters share this representation.
    """

    def __init__(
        self,
        latent_dim: int,
        n_corrections: int,
        hidden_dim: int,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_corrections = n_corrections
        self.layer = nn.Linear(latent_dim + n_corrections, hidden_dim)
        self.film = FiLM(n_corrections, hidden_dim) if n_corrections else None
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)
        self.apply(_orthogonal_init)

    def forward(
        self, latent: torch.Tensor, corrections: torch.Tensor | None = None
    ) -> torch.Tensor:
        """:returns: ``[samples, hidden_dim]``."""
        if self.n_corrections:
            if corrections is None:
                raise ValueError(
                    f"this decoder was built for {self.n_corrections} correction "
                    f"variable(s) but none were supplied"
                )
            h = torch.cat((latent, corrections.float()), dim=1)
        else:
            h = latent

        h = self.activation(self.layer(h))
        if self.film is not None:
            h = self.film(h, corrections)
        return self.dropout(h)


class LibrarySizeEncoder(nn.Module):
    """A sample's total count to its own small latent space.

    Library size is the total number of reads for a sample, and it varies for
    reasons that have nothing to do with biology. Modelling it separately lets
    the rest of the model work on composition instead of absolute depth.

    The input is the log of the total, because library sizes span orders of
    magnitude.
    """

    def __init__(
        self,
        n_corrections: int,
        hidden_dim: int,
        latent_dim: int,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_corrections = n_corrections
        self.layer = nn.Linear(1 + n_corrections, hidden_dim)
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_log_variance = nn.Linear(hidden_dim, latent_dim)
        self.film = FiLM(n_corrections, hidden_dim) if n_corrections else None
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)
        self.apply(_orthogonal_init)

    def forward(
        self, log_library_size: torch.Tensor, corrections: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = log_library_size.reshape(-1, 1)
        if self.n_corrections:
            if corrections is None:
                raise ValueError("correction variables were expected but not supplied")
            h = torch.cat((h, corrections.float()), dim=1)

        h = self.activation(self.layer(h))
        if self.film is not None:
            h = self.film(h, corrections)
        h = self.dropout(h)
        return self.to_mean(h), self.to_log_variance(h)


class LibrarySizeDecoder(nn.Module):
    """A library-size latent value back to a predicted log library size."""

    def __init__(
        self,
        latent_dim: int,
        n_corrections: int,
        hidden_dim: int,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.n_corrections = n_corrections
        self.layer = nn.Linear(latent_dim + n_corrections, hidden_dim)
        self.to_log_library_size = nn.Linear(hidden_dim, 1)
        self.film = FiLM(n_corrections, hidden_dim) if n_corrections else None
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.apply(_orthogonal_init)

    def forward(
        self, latent: torch.Tensor, corrections: torch.Tensor | None = None
    ) -> torch.Tensor:
        """:returns: predicted log library size, ``[samples, 1]``."""
        h = latent
        if self.n_corrections:
            if corrections is None:
                raise ValueError("correction variables were expected but not supplied")
            h = torch.cat((latent, corrections.float()), dim=1)

        h = self.activation(self.layer(h))
        if self.film is not None:
            h = self.film(h, corrections)
        return self.to_log_library_size(h)
