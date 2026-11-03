"""Turning checked count data into the tensors a model takes.

Keeps two count matrices apart, and the distinction matters:

- **inputs** are what the model sees. With masking on, some entries have been
  set to zero on purpose.
- **targets** are what it is scored against, always the original counts.

Version 0.1 kept both as well, under the names ``Xs`` and ``Rs``, but chose
between them inside the training loop with a test on the task name, so a run
could be scored against the wrong one depending on a string.

A third thing travels with a masked split: **how much of each sample's total
the hiding took away**. The model scales everything it predicts for a sample
by that sample's total count, and read off the inputs that total is short by
the hidden fraction while the targets are not. ``library_scale`` is the factor
that repairs it, derived from the mask whenever there is one, and finding 56
is what happened without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import torch

from preffect.config import CorrectionVariable
from preffect.data.graph import induce_subgraph, load_adjacency
from preffect.data.masking import Mask, apply_mask
from preffect.errors import DataError

__all__ = ["TensorData", "encode_corrections", "to_tensors"]


@dataclass(frozen=True, slots=True)
class TensorData:
    """One split, ready for a model."""

    inputs: torch.Tensor
    targets: torch.Tensor
    corrections: torch.Tensor | None = None
    batch_labels: torch.Tensor | None = None
    mask: Mask | None = None
    #: The sample-sample network as ``[2, edges]``, both directions, or None
    #: for the simple model. An edge list rather than a square matrix because
    #: these networks are very sparse; see data/graph.py.
    edge_index: torch.Tensor | None = None
    #: What each sample's observed total has to be multiplied by to estimate
    #: the total it would have had with nothing hidden, as ``[samples, 1]``.
    #: Filled in from the mask when there is one and left None when there is
    #: not, so a split with nothing hidden is untouched. See
    #: :meth:`~preffect.data.masking.Mask.library_scale` and finding 56.
    library_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.inputs.shape != self.targets.shape:
            raise DataError(
                f"inputs are shaped {tuple(self.inputs.shape)} and targets "
                f"{tuple(self.targets.shape)}; they have to match"
            )
        for name in ("corrections", "batch_labels", "library_scale"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != self.inputs.shape[0]:
                raise DataError(
                    f"{name} has {value.shape[0]} rows but there are "
                    f"{self.inputs.shape[0]} samples"
                )
        # Derived here rather than at every call site, so that an experiment
        # which builds its own TensorData with a mask gets the correction
        # without being edited. A scale supplied by hand wins, which is how
        # ``select`` carries one into a batch: a batch has no mask of its own.
        if self.mask is not None and self.library_scale is None:
            scale = torch.from_numpy(self.mask.library_scale()).to(
                device=self.inputs.device, dtype=self.inputs.dtype
            )
            if scale.shape[0] != self.inputs.shape[0]:
                raise DataError(
                    f"the mask covers {scale.shape[0]} samples but there are "
                    f"{self.inputs.shape[0]}"
                )
            object.__setattr__(self, "library_scale", scale)

    @property
    def n_samples(self) -> int:
        return self.inputs.shape[0]

    @property
    def n_genes(self) -> int:
        return self.inputs.shape[1]

    @property
    def has_network(self) -> bool:
        return self.edge_index is not None

    def select(self, positions: np.ndarray | torch.Tensor) -> TensorData:
        """One mini-batch, as a smaller :class:`TensorData`.

        When a network is present the subgraph is induced on the batch, so an
        edge survives only when both endpoints are in it. Most are not; see
        data/graph.py.

        The mask does not come along, because it describes the whole split and
        a batch is a handful of its rows. The library-size correction does, as
        the rows of it this batch covers, which is why it is a field of its own
        rather than something recomputed from the mask.
        """
        # On the data's device. Built on the CPU regardless, as it was until
        # finding 45, the subgraph induction below indexed a CPU table with a
        # CUDA edge_index and the first batch of any graph model on a GPU
        # raised a device mismatch.
        index = torch.as_tensor(positions, dtype=torch.long, device=self.inputs.device)
        return TensorData(
            inputs=self.inputs[index],
            targets=self.targets[index],
            corrections=None if self.corrections is None else self.corrections[index],
            batch_labels=(
                None if self.batch_labels is None else self.batch_labels[index]
            ),
            edge_index=(
                None
                if self.edge_index is None
                else induce_subgraph(self.edge_index, index, self.n_samples)
            ),
            library_scale=(
                None if self.library_scale is None else self.library_scale[index]
            ),
        )

    def adjacency_matrix(self) -> torch.Tensor:
        """The network for these samples as a square 0/1 matrix.

        Built on demand and only for a batch, where it is small. Holding the
        square form for a whole cohort would be 1.6 GB at 20,000 samples and
        almost entirely zeros.
        """
        if self.edge_index is None:
            raise DataError("this split carries no sample-sample network")
        dense = torch.zeros(
            (self.n_samples, self.n_samples),
            dtype=torch.float32,
            device=self.inputs.device,
        )
        if self.edge_index.numel():
            dense[self.edge_index[0], self.edge_index[1]] = 1.0
        return dense

    def to(self, device: str | torch.device) -> TensorData:
        """Return a copy with every tensor moved to ``device``.

        A new object rather than a move in place, because this class is frozen
        and because leaving the caller holding something whose contents have
        silently relocated is how a confusing device error happens three
        functions later.

        The mask is not a tensor and is carried across unchanged: it records
        which entries were hidden, which is a fact about the data rather than
        something the model computes with. The library-size correction derived
        from it is a tensor the model multiplies by, so it moves.
        """
        return TensorData(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            corrections=None if self.corrections is None else self.corrections.to(device),
            batch_labels=(
                None if self.batch_labels is None else self.batch_labels.to(device)
            ),
            mask=self.mask,
            edge_index=None if self.edge_index is None else self.edge_index.to(device),
            library_scale=(
                None if self.library_scale is None else self.library_scale.to(device)
            ),
        )


def encode_corrections(
    adata: ad.AnnData, variables: tuple[CorrectionVariable, ...]
) -> torch.Tensor | None:
    """Pull the correction variables out of the sample annotations.

    Categorical variables become integer codes, continuous ones stay as they
    are. Returns ``None`` when there are no correction variables, so the
    absence is explicit rather than an empty tensor that later code has to
    remember to check.

    :raises DataError: naming any variable the data does not carry, and listing
        what it does carry, since a misspelled column is the usual cause.
    """
    if not variables:
        return None

    missing = [v.name for v in variables if v.name not in adata.obs.columns]
    if missing:
        raise DataError(
            f"correction variable(s) {missing} are not in the sample annotations. "
            f"Available: {sorted(adata.obs.columns)}"
        )

    # Rows padded in for a patient missing this tissue carry no annotations at
    # all, so they are the likely cause of a missing value here and the
    # message says so rather than leaving someone looking at their own file.
    from preffect.data.tissues import ghost_rows

    n_padded = int(ghost_rows(adata).sum())
    padded_note = (
        f" {n_padded} row(s) here were padded in for patients with no sample in "
        f"this tissue, and a padded row carries no annotations, so a run that "
        f"pads cannot also use correction variables."
        if n_padded
        else ""
    )

    columns = []
    for variable in variables:
        values = adata.obs[variable.name]
        if variable.kind == "categorical":
            codes = values.astype("category").cat.codes.to_numpy()
            if (codes < 0).any():
                raise DataError(
                    f"correction variable {variable.name!r} has missing values. "
                    f"Fill them in, or give them a level of their own." + padded_note
                )
            columns.append(codes.astype(np.float32))
        else:
            numeric = values.to_numpy(dtype=np.float64)
            if np.isnan(numeric).any():
                raise DataError(
                    f"continuous correction variable {variable.name!r} has "
                    f"missing values." + padded_note
                )
            columns.append(numeric.astype(np.float32))

    return torch.from_numpy(np.column_stack(columns))


def to_tensors(
    adata: ad.AnnData,
    correction_variables: tuple[CorrectionVariable, ...] = (),
    mask: Mask | None = None,
    batch_column: str | None = "batch",
    with_network: bool = False,
) -> TensorData:
    """Build the tensors for one split.

    :param mask: entries to hide from the model. The targets keep the original
        values, so the model is scored on what it should have recovered. The
        library-size correction for the hiding is derived from it here, so
        nothing downstream has to remember to ask for it.
    :param batch_column: sample annotation naming each sample's batch, used
        only by the batch centroid term. Ignored when the column is absent.
    :param with_network: read the sample-sample network, which the single and
        full models need. Asking for it when it is absent is an error rather
        than a silent fallback: version 0.1 quietly turned a request for a
        graph model into a plain one, which is finding 2.
    """
    counts = adata.X
    dense = np.asarray(counts.todense()) if hasattr(counts, "todense") else np.asarray(counts)
    targets = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))

    if mask is None:
        inputs = targets.clone()
    else:
        inputs = torch.from_numpy(
            np.ascontiguousarray(apply_mask(dense, mask), dtype=np.float32)
        )

    batch_labels = None
    if batch_column and batch_column in adata.obs.columns:
        codes = adata.obs[batch_column].astype("category").cat.codes.to_numpy()
        batch_labels = torch.from_numpy(codes.astype(np.int64))

    return TensorData(
        inputs=inputs,
        targets=targets,
        corrections=encode_corrections(adata, correction_variables),
        batch_labels=batch_labels,
        mask=mask,
        edge_index=load_adjacency(adata) if with_network else None,
    )
