"""The full model: several matched tissues from the same patients.

The single model reads one profile and its network. The full model reads
several, a tumour and the normal tissue beside it for instance, matched patient
by patient, and uses them together.

## What it predicts, which is narrower than it sounds

**It predicts the first tissue's counts only.** The others inform that
prediction without being predicted themselves. Version 0.1 does the same: its
likelihood is built against tissue one's counts and tissue one's library size,
whatever the tissue count. The paper describes this as using matched tissues to
stabilise a profile, which is exactly what it is, but the phrase "multiple
expression profiles" can read as though all of them were being modelled
jointly. They are not.

Every tissue does get its own network reconstructed, so the objective has one
network term per tissue and one expression term in total.

## How the tissues are combined

Each tissue is encoded separately, producing one expression summary and one
network summary per tissue. The expression summaries are then concatenated and
squeezed back down to the size of one, and that combined summary is what the
count heads read. So information crosses between tissues at exactly one point,
after encoding and before decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from preffect.config import ModelConfig, ModelKind
from preffect.errors import ConfigError
from preffect.graph_layers import AdjacencyDecoder, GraphCountEncoder
from preffect.model import ModelOutput, SimpleModel

__all__ = ["FullModelOutput", "FullModel"]


@dataclass(frozen=True, slots=True)
class FullModelOutput:
    """The target tissue's counts, plus one network result per tissue."""

    counts_part: ModelOutput
    #: One entry per tissue, in the order the tissues were given.
    network_latent_mean: list[torch.Tensor]
    network_latent_log_variance: list[torch.Tensor]
    network_latent: list[torch.Tensor]
    predicted_network: list[torch.Tensor]

    def __getattr__(self, name: str):
        return getattr(self.counts_part, name)

    @property
    def n_tissues(self) -> int:
        return len(self.predicted_network)


class FullModel(SimpleModel):
    """Several matched tissues, each with its own sample-sample network."""

    def __init__(self, n_genes: int, config: ModelConfig, n_tissues: int):
        if config.kind is not ModelKind.FULL:
            raise ConfigError(
                f"FullModel needs kind 'full', got {config.kind.value!r}. Version "
                f"0.1 rewrote requests like this instead of refusing them, which "
                f"is finding 2."
            )
        if n_tissues < 2:
            raise ConfigError(
                f"the full model combines matched tissues, so it needs at least 2; "
                f"got {n_tissues}. Use 'single' for one tissue with a network."
            )
        super().__init__(n_genes, config)

        self.n_tissues = n_tissues
        n_corrections = len(config.correction_variables)

        def graph_encoder() -> GraphCountEncoder:
            return GraphCountEncoder(
                n_genes=n_genes,
                n_corrections=n_corrections,
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
                attention_heads=config.attention_heads,
                leaky_relu_slope=config.leaky_relu_slope,
                dropout=config.dropout,
            )

        # One pair per tissue. The parent's plain count encoder is replaced,
        # since here the counts are always read through a network.
        self.encoder = None
        self.encoders = nn.ModuleList([graph_encoder() for _ in range(n_tissues)])
        self.network_encoders = nn.ModuleList([graph_encoder() for _ in range(n_tissues)])
        self.network_decoders = nn.ModuleList(
            [
                AdjacencyDecoder(
                    latent_dim=config.latent_dim,
                    n_corrections=n_corrections,
                    hidden_dim=config.hidden_dim,
                    leaky_relu_slope=config.leaky_relu_slope,
                    dropout=config.dropout,
                )
                for _ in range(n_tissues)
            ]
        )

        # The one place information crosses between tissues: concatenate the
        # per-tissue expression summaries and squeeze back to one summary's
        # worth before decoding.
        self.combine = nn.Linear(config.latent_dim * n_tissues, config.latent_dim)

    def forward(
        self,
        counts: list[torch.Tensor],
        corrections: torch.Tensor | None = None,
        edge_index: list[torch.Tensor] | None = None,
        library_scale: torch.Tensor | None = None,
    ) -> FullModelOutput:
        """Run one batch of matched tissues through the model.

        :param counts: one ``[samples, genes]`` tensor per tissue, aligned by row.
        :param library_scale: ``[samples, 1]`` for the **target** tissue, the
            only one whose counts are predicted, and so the only one whose
            total sets a scale. None when nothing is hidden. See
            :meth:`~preffect.model.SimpleModel._library_size`.
        """
        if edge_index is None:
            raise ConfigError(
                "the full model needs one sample-sample network per tissue and "
                "none were given. Read the data with with_network=True."
            )
        if not (len(counts) == len(edge_index) == self.n_tissues):
            raise ConfigError(
                f"this model was built for {self.n_tissues} tissues but was given "
                f"{len(counts)} count matrices and {len(edge_index)} networks"
            )
        for index, tissue in enumerate(counts, start=1):
            if tissue.shape[1] != self.n_genes:
                raise ValueError(
                    f"tissue {index} has {tissue.shape[1]} genes, but this model "
                    f"was built for {self.n_genes}"
                )

        cfg = self.config
        summaries, net_means, net_vars, net_latents, predicted = [], [], [], [], []

        for i in range(self.n_tissues):
            mean, log_variance = self.encoders[i](counts[i], edge_index[i], corrections)
            mean = mean.clamp(-cfg.max_latent_mean, cfg.max_latent_mean)
            log_variance = log_variance.clamp(
                -cfg.max_log_variance, cfg.max_log_variance
            )
            summaries.append(self.reparameterise(mean, log_variance))
            if i == 0:
                target_mean, target_log_variance = mean, log_variance

            n_mean, n_log_variance = self.network_encoders[i](
                counts[i], edge_index[i], corrections
            )
            n_mean = n_mean.clamp(-cfg.max_latent_mean, cfg.max_latent_mean)
            n_log_variance = n_log_variance.clamp(
                -cfg.max_log_variance, cfg.max_log_variance
            )
            n_latent = self.reparameterise(n_mean, n_log_variance)

            net_means.append(n_mean)
            net_vars.append(n_log_variance)
            net_latents.append(n_latent)
            predicted.append(self.network_decoders[i](n_latent, corrections))

        combined = self.combine(torch.cat(summaries, dim=1))

        # The target tissue's counts, its latent summary built from every
        # tissue, and its own library-size correction. The parent's version of
        # this step does exactly what is wanted here, so it is used rather than
        # copied: an identical copy lived here until 2026-11-02, and threading
        # the library correction through the models is what found it.
        counts_part = self._finish_counts(
            counts[0],
            corrections,
            combined,
            target_mean,
            target_log_variance,
            library_scale=library_scale,
        )

        return FullModelOutput(
            counts_part=counts_part,
            network_latent_mean=net_means,
            network_latent_log_variance=net_vars,
            network_latent=net_latents,
            predicted_network=predicted,
        )
