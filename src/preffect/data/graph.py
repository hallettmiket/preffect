"""The sample-sample network: reading it, and cutting it down to a mini-batch.

The graph models attach a network saying which samples resemble each other. It
is stored in the count file under ``obsm['sample_sample_adj']`` as a square
matrix, one row and column per sample, with a one where two samples are linked.

## Edges are kept as a list, not a square matrix

A square matrix of 20,000 samples is 1.6 GB and almost entirely zeros: these
networks have three or four neighbours per sample. The edge list is a few
hundred kilobytes. The square form is rebuilt only for the handful of samples
in a mini-batch, where it is small.

## Most edges do not survive a mini-batch, and that is reported

An edge is usable only when **both** of its endpoints are in the same batch,
and the share that survives is the batch size divided by the sample count. On a
cohort of a few thousand that is a small fraction; on this project's real
datasets, which run from 48 to 528 samples, it is between 38% and all of them.
Version 0.1 did the same thing silently either way. Here :func:`edge_survival`
measures it so a run can say so.

Finding 22 in ``how_this_project_breaks.md``; whether to batch the graph
properly instead is issue #12.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import torch

from preffect.errors import DataError

__all__ = [
    "ADJACENCY_KEY",
    "EdgeSurvival",
    "has_adjacency",
    "load_adjacency",
    "induce_subgraph",
    "edge_survival",
]

#: Where version 0.1 stores the network, and where version 0.2 looks.
ADJACENCY_KEY = "sample_sample_adj"


def has_adjacency(adata: ad.AnnData) -> bool:
    """Does this file carry a sample-sample network at all?

    Says only whether the entry is present, not whether it contains any edges.
    Every one of the nine datasets version 0.1 shipped carries an identity
    matrix, which is an entry with every sample linked to itself and nothing
    else, and :func:`load_adjacency` drops self-links, so the edge list comes
    back empty. This returns True for all of them. Finding 35, which is how
    that was discovered for eight of them: a graph model trained for hours over
    a graph with no edges, and the loss fell the whole time. Finding 54 found
    the ninth was the same identity with its rows subset, and why: a bug in
    the script that built them all.
    """
    return ADJACENCY_KEY in adata.obsm


def load_adjacency(adata: ad.AnnData, split: str = "data") -> torch.Tensor:
    """Read the network as an edge list of shape ``[2, edges]``.

    Both directions of every edge are included, which is what a graph attention
    layer expects. Self-links are dropped: every sample attends to itself
    anyway, and counting them would flatter any measure of how well the network
    is reconstructed.

    :raises DataError: if the network is missing or not square.
    """
    if not has_adjacency(adata):
        raise DataError(
            f"{split} has no sample-sample network under obsm[{ADJACENCY_KEY!r}], "
            f"which the single and full models need. Use the simple model, or "
            f"add one. Present: {sorted(adata.obsm) or '(nothing)'}"
        )

    matrix = adata.obsm[ADJACENCY_KEY]
    dense = np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)

    n = adata.n_obs
    if dense.shape != (n, n):
        raise DataError(
            f"{split}'s network is shaped {dense.shape} but there are {n} samples; "
            f"it has to be square with one row per sample"
        )

    dense = np.array(dense)
    self_links = int(np.count_nonzero(np.diag(dense)))
    np.fill_diagonal(dense, 0)
    rows, cols = np.nonzero(dense)

    if rows.size == 0:
        raise DataError(
            f"{split}'s sample-sample network has no edges between different "
            f"samples"
            + (f" (only {self_links} self-links)" if self_links else "")
            + f". A graph model over {n} unconnected samples has nothing to "
            f"attend to: it costs about ten times the parameters and twenty "
            f"times the time of the simple model and cannot use any of it. "
            f"Use the simple model, or supply a network with real edges. "
            f"Every dataset version 0.1 shipped carries an identity matrix "
            f"here, so this is the common case, not a rare one. See findings "
            f"35 and 54."
        )

    return torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))


def induce_subgraph(
    edge_index: torch.Tensor, positions: torch.Tensor, n_total: int
) -> torch.Tensor:
    """Keep the edges whose endpoints are both in this batch, renumbered locally.

    :param positions: the samples in the batch, as positions into the whole
        split.
    :returns: ``[2, kept]`` with indices in ``0 .. len(positions) - 1``, ready
        for a graph layer that knows nothing about the samples left out.
    """
    # Everything on edge_index's device. The table and the positions were
    # built on the CPU whatever the edges' device, which no CPU run could
    # notice and every GPU run fell over on. Finding 45.
    device = edge_index.device
    positions = torch.as_tensor(positions, dtype=torch.long, device=device)
    local = torch.full((n_total,), -1, dtype=torch.long, device=device)
    local[positions] = torch.arange(len(positions), device=device)

    both_present = (local[edge_index[0]] >= 0) & (local[edge_index[1]] >= 0)
    kept = edge_index[:, both_present]
    return torch.stack([local[kept[0]], local[kept[1]]])


@dataclass(frozen=True, slots=True)
class EdgeSurvival:
    """How much of the network a batching scheme leaves usable."""

    total_edges: int
    kept_edges: int
    n_samples: int
    batch_size: int

    @property
    def fraction(self) -> float:
        return self.kept_edges / self.total_edges if self.total_edges else 0.0

    def __str__(self) -> str:
        return (
            f"{self.kept_edges} of {self.total_edges} edges usable "
            f"({self.fraction:.1%}) at a batch size of {self.batch_size} "
            f"over {self.n_samples} samples"
        )


def edge_survival(
    edge_index: torch.Tensor,
    batches: list[np.ndarray] | list[torch.Tensor],
    n_total: int,
) -> EdgeSurvival:
    """Count how many edges survive a particular division into batches.

    Counts each undirected edge once, so the number matches what a person would
    get by counting lines in the network.
    """
    undirected = edge_index[:, edge_index[0] < edge_index[1]]
    total = undirected.shape[1]

    kept = sum(
        induce_subgraph(undirected, torch.as_tensor(positions), n_total).shape[1]
        for positions in batches
    )
    sizes = [len(b) for b in batches]
    return EdgeSurvival(
        total_edges=total,
        kept_edges=kept,
        n_samples=n_total,
        batch_size=max(sizes) if sizes else 0,
    )
