"""Choosing which genes and which samples to use.

**Genes are chosen once, from the training data, and then applied by name.**
Version 0.1 chose them independently for every split: each one ranked its own
genes by variability and kept its own top N. On the simple vignette, asking for
the twenty most variable genes gave training and validation sets that disagreed
about nine of the twenty, so column *j* of one matrix and column *j* of the
other referred to different genes. The shape check compared widths, which
matched, so nothing caught it. That is defect 18 in
``how_this_project_breaks.md``.

Applying by name rather than by position is the second half of the fix. Two
files can hold the same genes in a different order, and selecting positions
computed from one file would then silently take the wrong columns from another.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np

from preffect.errors import DataError

__all__ = [
    "GeneSelection",
    "rank_genes_by_variability",
    "choose_genes",
    "apply_gene_selection",
    "choose_samples",
]


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)


@dataclass(frozen=True, slots=True)
class GeneSelection:
    """The genes a run uses, decided once and recorded.

    Carrying the names rather than positions is what makes this safe to apply
    to a second file.
    """

    names: tuple[str, ...]
    #: How they were chosen, kept so a run can say what it did.
    basis: str = "interquartile range, most variable first"

    def __len__(self) -> int:
        return len(self.names)


def rank_genes_by_variability(adata: ad.AnnData) -> list[str]:
    """Rank genes most variable first, by interquartile range.

    The interquartile range is the spread of the middle half of a gene's counts
    across samples: the 75th percentile minus the 25th. It is used in
    preference to the variance because a single extreme count, which fRNA-seq
    data has plenty of, moves the variance a long way and the interquartile
    range hardly at all.
    """
    counts = _dense(adata.X).astype(float)
    if counts.size == 0:
        return []
    upper, lower = np.nanpercentile(counts, [75, 25], axis=0)
    spread = upper - lower
    # Ties broken by position, so the result does not depend on sort internals.
    order = np.lexsort((np.arange(len(spread)), -spread))
    return [str(adata.var_names[i]) for i in order]


def choose_genes(
    train: ad.AnnData,
    n_genes: int | None,
    force_genes=(),
) -> GeneSelection:
    """Decide the genes for a whole run, from the training data alone.

    :param n_genes: how many to keep, or ``None`` to keep all of them.
    :param force_genes: genes to keep whatever their variability, for example a
        published panel. Those present are placed first. Accepts plain names, or
        a :class:`~preffect.gene_sets.GeneSet`, which carries both gene symbols
        and Ensembl identifiers and is matched whichever way this data names its
        genes.

    **A panel that matches nothing raises rather than doing nothing.** Names
    absent from the data are still ignored individually, because a panel
    routinely names genes an assay did not measure. But *none* of them matching
    means the wrong naming was passed, and version 0.1 is the reason this is an
    error: its ``force_pam50`` setting matched an Ensembl list against
    symbol-named data, found nothing, and reordered the ranking by an empty
    list, which leaves it exactly as it was. Three of its own PAM50 notebooks
    ran that way. Finding 33.
    """
    from preffect.gene_sets import GeneSet, match_genes

    available = {str(name) for name in train.var_names}

    if isinstance(force_genes, GeneSet):
        _, forced, _ = match_genes(force_genes, [str(n) for n in train.var_names])
        wanted, description = len(force_genes), force_genes.name
    else:
        forced = [g for g in force_genes if g in available]
        wanted, description = len(tuple(force_genes)), "the genes named"

    if wanted and not forced:
        first = str(next(iter(train.var_names), "?"))
        raise DataError(
            f"none of the {wanted} gene(s) in {description} are in this data, so "
            f"forcing them would silently do nothing. This data names its first "
            f"gene {first!r}; check that the panel uses the same naming. Pass a "
            f"GeneSet to have both gene symbols and Ensembl identifiers tried."
        )
    ranked = [g for g in rank_genes_by_variability(train) if g not in set(forced)]
    ordered = forced + ranked

    if n_genes is None:
        kept = ordered
        basis = "all genes"
    else:
        if n_genes < 1:
            raise DataError(f"cannot keep {n_genes} genes; ask for at least 1")
        kept = ordered[:n_genes]
        basis = "interquartile range, most variable first"
        if forced:
            basis = f"{len(forced)} named gene(s) first, then " + basis

    return GeneSelection(names=tuple(kept), basis=basis)


def apply_gene_selection(
    tissues: list[ad.AnnData],
    selection: GeneSelection,
    split: str = "data",
) -> list[ad.AnnData]:
    """Restrict every tissue to the chosen genes, matching on name.

    :raises DataError: if any chosen gene is absent, naming the ones missing.
    """
    out = []
    for index, adata in enumerate(tissues, start=1):
        present = {str(name) for name in adata.var_names}
        missing = [g for g in selection.names if g not in present]
        if missing:
            shown = ", ".join(missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise DataError(
                f"{split} tissue {index} is missing {len(missing)} of the "
                f"{len(selection)} genes chosen from the training data: {shown}{more}. "
                f"Every split has to carry the same genes."
            )
        out.append(adata[:, list(selection.names)])
    return out


def choose_samples(
    n_available: int,
    n_wanted: int | None,
    rng: np.random.Generator,
    replace: bool = False,
) -> np.ndarray:
    """Pick which samples to use, as positions into the data.

    Returns every sample in order when ``n_wanted`` is ``None``, so the common
    case involves no randomness at all.

    :raises DataError: if more samples are asked for than exist and sampling
        with replacement was not requested. Version 0.1 turned replacement on
        by itself in that situation, which silently changed the experiment:
        drawing 500 samples with replacement from 100 is not the same study as
        having 500.
    """
    if n_wanted is None:
        return np.arange(n_available)
    if n_wanted < 1:
        raise DataError(f"cannot select {n_wanted} samples; ask for at least 1")
    if n_wanted > n_available and not replace:
        raise DataError(
            f"asked for {n_wanted} samples but only {n_available} exist. To "
            f"draw repeats deliberately, ask for sampling with replacement; "
            f"note that this duplicates samples rather than adding information."
        )
    return rng.choice(n_available, size=n_wanted, replace=replace)
