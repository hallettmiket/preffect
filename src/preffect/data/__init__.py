"""Reading count data and checking it is usable."""

from __future__ import annotations

from preffect.data.batching import batch_indices, n_batches
from preffect.data.checks import (
    SplitOverlap,
    ValidationRepeatsTrainingWarning,
    check_tissues,
    compare_splits,
    problems_with_counts,
    warn_if_validation_repeats_training,
)
from preffect.data.loading import (
    TISSUE_FILE_PATTERN,
    find_tissue_files,
    load_tissues,
)
from preffect.data.masking import MCAR, Mask, apply_mask, mcar_mask
from preffect.data.selection import (
    GeneSelection,
    apply_gene_selection,
    choose_genes,
    choose_samples,
    rank_genes_by_variability,
)

__all__ = [
    "MCAR",
    "TISSUE_FILE_PATTERN",
    "GeneSelection",
    "Mask",
    "SplitOverlap",
    "ValidationRepeatsTrainingWarning",
    "apply_gene_selection",
    "apply_mask",
    "batch_indices",
    "check_tissues",
    "choose_genes",
    "choose_samples",
    "compare_splits",
    "find_tissue_files",
    "load_tissues",
    "mcar_mask",
    "n_batches",
    "problems_with_counts",
    "rank_genes_by_variability",
    "warn_if_validation_repeats_training",
]
