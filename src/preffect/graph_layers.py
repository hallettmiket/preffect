"""Network pieces that use the sample-sample network.

The single and full models attach a network saying which samples resemble each
other, and let a **graph attention layer** pass information along its edges. A
sample's representation is then built partly from its neighbours, which is
meant to stabilise profiles that are individually noisy.

Requires the optional ``graph`` extra::

    uv pip install -e ".[graph]"

## Two things about these networks, both measured

They are **very sparse**: three or four neighbours per sample across every
dataset in this project. And version 0.1 induces the subgraph on each
mini-batch, so an edge survives only when both of its endpoints land in the
same batch, which means the share of the network visible in an epoch is the
batch size divided by the sample count.

That share only gets small on a large cohort. This project's real datasets run
from 48 to 528 samples, so at version 0.1's default batch of 200 they see
between 38% and all of their network each epoch. An earlier version of this
paragraph said "about a tenth at the published settings", which came from a
synthetic 2,000-sample dataset used in one scratch notebook. Finding 22 in
``how_this_project_breaks.md``, issue #12.

## The adjacency decoder is not a faithful port, and could not be

Version 0.1 reconstructs the network with a linear layer whose output width is
the **mini-batch size**, so the model's parameter count follows a training
setting: 7,400 parameters at a batch of 50 and 15,050 at a batch of 200. It can
only ever process batches of exactly the size it was built for, which is why
version 0.1 had to discard the last short batch of every epoch. Keeping every
sample was decided and tested earlier, and the two are incompatible.

So this uses the conventional graph autoencoder decoder instead: project the
latent values, then take the inner product between every pair and squash it to
a probability. An edge is predicted where two samples have similar latent
representations, which is what an edge is supposed to mean. It works at any
batch size and has fewer parameters. Recorded as finding 23 and in
``results_affecting_changes.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from preffect.errors import PreffectError
from preffect.layers import FiLM, _orthogonal_init, scale_counts

__all__ = ["require_graph_support", "GraphCountEncoder", "AdjacencyDecoder"]


def require_graph_support():
    """Import the graph attention layer, or explain how to get it.

    The import is wrapped so that one warning does not reach the user.
    torch-geometric 2.8.1 calls ``torch.jit.script`` while it is being imported
    (``nn/pool/select/base.py``), and PyTorch 2.14 answers with a
    ``FutureWarning`` that the function is deprecated. That is torch-geometric's
    to fix. It fires once, at import, and never again on construction or use,
    and it told every person training a graph model about a deprecation in a
    library they had not heard of. The filter matches that one message only
    and lasts for this import only; a test runs the import under
    ``-W error::FutureWarning`` so the warning cannot come back unnoticed.
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"`torch\.jit\.script` is deprecated",
            )
            from torch_geometric.nn.conv import GATv2Conv
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise PreffectError(
            "the single and full models need torch-geometric, which is an "
            'optional dependency. Install it with: pip install "preffect[graph]"'
        ) from exc
    return GATv2Conv


class GraphCountEncoder(nn.Module):
    """Counts plus a sample-sample network, to a latent mean and log variance.

    The same class serves both latent spaces the graph models keep: one
    summarising expression and one summarising network position. Version 0.1
    used one class for both too, which is the part of its design that was right.
    """

    def __init__(
        self,
        n_genes: int,
        n_corrections: int,
        hidden_dim: int,
        latent_dim: int,
        attention_heads: int = 8,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        gat = require_graph_support()

        self.n_corrections = n_corrections
        input_dim = n_genes + n_corrections
        wide = hidden_dim * attention_heads

        self.attention = gat(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=attention_heads,
            negative_slope=leaky_relu_slope,
        )
        self.narrow = nn.Linear(wide, hidden_dim)
        # Version 0.1 calls this layer3_test. It is in the live path, not a
        # leftover, so it is ported; the name is not.
        self.refine = nn.Linear(hidden_dim, hidden_dim)

        self.to_mean = nn.Linear(hidden_dim + n_corrections, latent_dim)
        self.to_log_variance = nn.Linear(hidden_dim + n_corrections, latent_dim)

        self.film1 = FiLM(n_corrections, wide) if n_corrections else None
        self.film2 = FiLM(n_corrections, hidden_dim) if n_corrections else None

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)

        self.apply(_orthogonal_init)

    def scale_input(self, counts: torch.Tensor) -> torch.Tensor:
        """The scale the attention layer sees: ``log(1 + counts)``, as the simple model's encoder.

        Measured on the simple model (``exp/10``); the graph models get it on
        the same reasoning without a measurement of their own, which the
        decision record says and an issue tracks.
        """
        return scale_counts(counts)

    def combine_with_corrections(
        self, counts: torch.Tensor, corrections: torch.Tensor | None
    ) -> torch.Tensor:
        """Attach the correction variables to the counts, if there are any.

        Correction variables are things recorded about a sample that are not
        expression and that the model should account for, such as which batch
        it was sequenced in. They are concatenated along the feature axis, so
        the encoder sees one wider row per sample. `exp/02` tested embedding
        them instead and found no benefit, which is why this is still a
        concatenation.

        :raises ValueError: if the encoder was built expecting correction
            variables and none were supplied. Version 0.1 carried on with
            whatever happened to be there, which is the habit this package
            exists to drop.
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
        self, hidden: torch.Tensor, corrections: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """No activation, matching the simple model after ``exp/01``.

        ``exp/01`` tested this on the simple model only. Applying the same
        answer here is a consistency judgement rather than a measurement, and
        issue #4 covers repeating it.
        """
        if self.n_corrections and corrections is not None:
            hidden = torch.cat((hidden, corrections.float()), dim=1)
        return self.to_mean(hidden), self.to_log_variance(hidden)

    def forward(
        self,
        counts: torch.Tensor,
        edge_index: torch.Tensor,
        corrections: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:param edge_index: ``[2, edges]``, the network for this batch."""
        h = self.combine_with_corrections(self.scale_input(counts), corrections)

        h = torch.nn.functional.elu(self.attention(h, edge_index))
        if self.film1 is not None:
            h = self.film1(h, corrections)
        h = self.dropout1(h)

        h = torch.nn.functional.elu(self.narrow(h))
        if self.film2 is not None:
            h = self.film2(h, corrections)
        h = self.dropout2(h)

        h = self.dropout3(torch.nn.functional.elu(self.refine(h)))
        return self.project_to_latent(h, corrections)


class AdjacencyDecoder(nn.Module):
    """Latent values back to a predicted sample-sample network.

    Projects the latent values, then scores every pair by inner product and
    squashes to a probability, so an edge is predicted where two samples sit
    close together in the latent space. That is what an edge means, and it is
    the conventional graph autoencoder decoder.

    Deliberately not version 0.1's arrangement, which used a linear layer of
    width equal to the mini-batch size. See this module's docstring.
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
        self.refine = nn.Linear(hidden_dim, hidden_dim)
        self.film = FiLM(n_corrections, hidden_dim) if n_corrections else None
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.apply(_orthogonal_init)

    def forward(
        self, latent: torch.Tensor, corrections: torch.Tensor | None = None
    ) -> torch.Tensor:
        """:returns: ``[samples, samples]`` edge probabilities for this batch."""
        h = latent
        if self.n_corrections:
            if corrections is None:
                raise ValueError("correction variables were expected but not supplied")
            h = torch.cat((latent, corrections.float()), dim=1)

        h = self.activation(self.layer(h))
        if self.film is not None:
            h = self.film(h, corrections)
        h = self.dropout1(h)
        h = self.dropout2(self.activation(self.refine(h)))

        return torch.sigmoid(h @ h.transpose(0, 1))
