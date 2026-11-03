"""The single model: counts plus a sample-sample network.

The simple model looks at each sample alone. The single model adds a network
saying which samples resemble each other, and lets a graph attention layer
build each sample's representation partly from its neighbours. The hope is that
a noisy profile is stabilised by the company it keeps.

## Two latent spaces, not one

The model keeps a separate summary for each of the two things it is asked to
reproduce: one for the counts, one for the network. Version 0.1 does the same,
and the reason is that they answer different questions. A latent space good at
predicting who your neighbours are need not be good at predicting your counts.

So a forward pass runs the graph encoder twice over the same input, producing
an expression summary and a network summary, and the network summary is the one
the adjacency decoder reads.

## What the objective gains

Two terms beyond the simple model's: how well the network is reproduced, and
how far the network summary has drifted from the prior. Both only exist here,
so for a simple model they are absent from the objective rather than present
and zero. That distinction is why the loss is a list.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from preffect.config import ModelConfig, ModelKind
from preffect.errors import ConfigError
from preffect.graph_layers import AdjacencyDecoder, GraphCountEncoder
from preffect.model import ModelOutput, SimpleModel

__all__ = ["GraphModelOutput", "SingleModel"]


@dataclass(frozen=True, slots=True)
class GraphModelOutput:
    """A :class:`ModelOutput` plus what the network half produced.

    Composed rather than subclassed: the count half is exactly what the simple
    model produces, and saying so in the type keeps the loss from having to ask
    which kind of model it is looking at.
    """

    counts_part: ModelOutput
    network_latent_mean: torch.Tensor
    network_latent_log_variance: torch.Tensor
    network_latent: torch.Tensor
    predicted_network: torch.Tensor

    def __getattr__(self, name: str):
        """Read the count half's fields directly, so callers need not unwrap."""
        return getattr(self.counts_part, name)


class SingleModel(SimpleModel):
    """Counts and one sample-sample network, for one tissue."""

    def __init__(self, n_genes: int, config: ModelConfig):
        if config.kind is ModelKind.SIMPLE:
            raise ConfigError(
                "SingleModel was asked for with kind 'simple'. Use SimpleModel, or "
                "set kind to 'single'. Version 0.1 silently rewrote this request, "
                "which is finding 2."
            )
        # The parent builds the count encoder, the decoder, the parameter heads
        # and optionally the library pair. The count encoder is then replaced
        # with the graph one, since for this model the counts are read through
        # the network.
        super().__init__(n_genes, config)

        n_corrections = len(config.correction_variables)
        self.encoder = GraphCountEncoder(
            n_genes=n_genes,
            n_corrections=n_corrections,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            attention_heads=config.attention_heads,
            leaky_relu_slope=config.leaky_relu_slope,
            dropout=config.dropout,
        )
        self.network_encoder = GraphCountEncoder(
            n_genes=n_genes,
            n_corrections=n_corrections,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            attention_heads=config.attention_heads,
            leaky_relu_slope=config.leaky_relu_slope,
            dropout=config.dropout,
        )
        self.network_decoder = AdjacencyDecoder(
            latent_dim=config.latent_dim,
            n_corrections=n_corrections,
            hidden_dim=config.hidden_dim,
            leaky_relu_slope=config.leaky_relu_slope,
            dropout=config.dropout,
        )

    def forward(
        self,
        counts: torch.Tensor,
        corrections: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        library_scale: torch.Tensor | None = None,
    ) -> GraphModelOutput:
        """Run one batch through the model.

        :param edge_index: ``[2, edges]`` for this batch, local indices.
        :param library_scale: ``[samples, 1]``, what each sample's observed
            total is multiplied by when entries have been hidden from it;
            None when nothing is hidden. See
            :meth:`~preffect.model.SimpleModel._library_size`.
        """
        if edge_index is None:
            raise ConfigError(
                "the single model needs a sample-sample network and none was "
                "given. Read the data with with_network=True, or use the simple "
                "model."
            )
        if counts.shape[1] != self.n_genes:
            raise ValueError(
                f"this model was built for {self.n_genes} genes but was given "
                f"{counts.shape[1]}"
            )

        cfg = self.config

        mean, log_variance = self.encoder(counts, edge_index, corrections)
        mean = mean.clamp(-cfg.max_latent_mean, cfg.max_latent_mean)
        log_variance = log_variance.clamp(-cfg.max_log_variance, cfg.max_log_variance)
        latent = self.reparameterise(mean, log_variance)

        net_mean, net_log_variance = self.network_encoder(counts, edge_index, corrections)
        net_mean = net_mean.clamp(-cfg.max_latent_mean, cfg.max_latent_mean)
        net_log_variance = net_log_variance.clamp(
            -cfg.max_log_variance, cfg.max_log_variance
        )
        net_latent = self.reparameterise(net_mean, net_log_variance)

        counts_part = self._finish_counts(
            counts, corrections, latent, mean, log_variance, library_scale=library_scale
        )

        return GraphModelOutput(
            counts_part=counts_part,
            network_latent_mean=net_mean,
            network_latent_log_variance=net_log_variance,
            network_latent=net_latent,
            predicted_network=self.network_decoder(net_latent, corrections),
        )
