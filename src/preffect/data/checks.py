"""Checks on count data, which stop a run rather than noting a problem and continuing.

Version 0.1 had a function that looked like it validated its input. Every check
inside it was written as "raise an error, catch that same error, write it to a
log, carry on", and the function ended with an unconditional ``return True``.
The caller read::

    if not self.check_fidelity_of_anndata():
        raise ValueError('Something wrong with input data. Check log.')

which could never fire. Malformed input was reported to a log file nobody opens
and then used anyway. That is defect 16 in ``how_this_project_breaks.md``.

Here the checks collect every problem they find and raise once, so a researcher
with a malformed file is told everything that is wrong with it in one go rather
than discovering the faults one run at a time.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import anndata as ad
import numpy as np

from preffect.errors import DataError

__all__ = [
    "ValidationRepeatsTrainingWarning",
    "SplitOverlap",
    "problems_with_counts",
    "check_tissues",
    "compare_splits",
    "warn_if_validation_repeats_training",
]


class ValidationRepeatsTrainingWarning(UserWarning):
    """The validation split holds the same samples as the training split.

    Not an error. A cohort too small to divide has nowhere else to go, and that
    is exactly how version 0.1's real datasets came to have identical training
    and validation folders. It is a warning because the resulting validation
    loss is a training loss, and nothing in version 0.1 said so.
    """


def problems_with_counts(adata: ad.AnnData, label: str) -> list[str]:
    """Return every problem with one tissue's counts, as readable sentences.

    An empty list means the data is usable.
    """
    problems: list[str] = []

    if adata.X is None:
        return [f"{label}: has no expression matrix"]

    matrix = adata.X
    dense = np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)

    if dense.ndim != 2:
        return [f"{label}: expression matrix has {dense.ndim} dimensions, expected 2"]

    n_samples, n_genes = dense.shape
    if n_samples == 0:
        problems.append(f"{label}: no samples")
    if n_genes == 0:
        problems.append(f"{label}: no genes")

    if dense.size:
        if np.isnan(dense).any():
            count = int(np.isnan(dense).sum())
            problems.append(
                f"{label}: {count} missing value(s) in the counts. A count model "
                f"cannot use them; set them to 0 or drop the gene or sample."
            )
        finite = dense[np.isfinite(dense)]
        if finite.size and finite.min() < 0:
            problems.append(
                f"{label}: smallest count is {finite.min()}, and counts cannot be "
                f"negative. This usually means the matrix holds transformed values "
                f"rather than raw counts."
            )
        if finite.size and not np.all(np.equal(np.mod(finite, 1), 0)):
            problems.append(
                f"{label}: counts are not whole numbers. PREFFECT models counts, so "
                f"normalised or log-transformed values will not fit."
            )

    problems.extend(_repeated_names(adata.obs_names, "sample", label))
    problems.extend(_repeated_names(adata.var_names, "gene", label))

    return problems


def _repeated_names(names, kind: str, label: str) -> list[str]:
    """Report repeated names, naming them and saying what to do.

    **Found by running the check on real data.** It used to say only "gene names
    are not unique", which is true and useless: GSE167977, the largest dataset
    here, has 21 genes all called ``No_Match``, the placeholder its preparation
    wrote where a symbol lookup failed. Without the name and the count there is
    nothing to act on. Finding 34.

    This matters beyond tidiness because a gene name is the key everything else
    uses. Selection matches by name, so a repeated name selects several columns
    for one request and quietly changes the width of the matrix. Version 0.1
    built ``{gene: index for index, gene in enumerate(names)}``, which keeps the
    last occurrence and drops the rest without a word.
    """
    from collections import Counter

    counts = Counter(str(name) for name in names)
    repeated = {name: n for name, n in counts.items() if n > 1}
    if not repeated:
        return []

    worst = sorted(repeated.items(), key=lambda pair: -pair[1])
    shown = ", ".join(f"{name} x{n}" for name, n in worst[:3])
    more = f" and {len(worst) - 3} more" if len(worst) > 3 else ""
    total = sum(repeated.values())

    return [
        f"{label}: {len(repeated)} {kind} name(s) appear more than once, "
        f"covering {total} of {len(counts) + total - len(repeated)} columns "
        f"({shown}{more}). A {kind} name is the key selection matches on, so a "
        f"repeat silently pulls several columns for one request. Drop them, or "
        f"make the names unique before loading."
    ]


def check_tissues(tissues: list[ad.AnnData], split: str = "data") -> None:
    """Check every tissue, and report all problems at once.

    :param split: what to call this set in the message, for example "training".
    :raises DataError: if any tissue has a problem.
    """
    problems: list[str] = []
    for index, adata in enumerate(tissues, start=1):
        problems.extend(problems_with_counts(adata, f"{split} tissue {index}"))

    gene_sets = [tuple(t.var_names) for t in tissues]
    if len(tissues) > 1 and len(set(gene_sets)) > 1:
        sizes = [len(g) for g in gene_sets]
        problems.append(
            f"{split}: the tissues do not share the same genes, in the same order "
            f"(gene counts {sizes}). Matched tissues are combined gene by gene."
        )

    if problems:
        raise DataError(
            f"{len(problems)} problem(s) with the {split} data:\n  - "
            + "\n  - ".join(problems)
        )


@dataclass(frozen=True, slots=True)
class SplitOverlap:
    """How much two splits have in common."""

    n_train: int
    n_validation: int
    n_shared_names: int
    #: Of the samples sharing a name, how many also hold identical counts. This
    #: is the number that means anything. A shared name on its own does not:
    #: simulated data often numbers its samples '0', '1', '2', so two genuinely
    #: independent draws share every name while holding different counts. An
    #: earlier version of this check used names alone and reported the simple
    #: vignette, which is two independent draws, as overlapping.
    n_identical_rows: int

    @property
    def fraction_repeated(self) -> float:
        """How much of the training set reappears in validation, by value."""
        return self.n_identical_rows / self.n_train if self.n_train else 0.0

    @property
    def values_identical_on_shared(self) -> bool:
        """Every sample sharing a name also holds identical counts."""
        return self.n_shared_names > 0 and self.n_identical_rows == self.n_shared_names

    @property
    def is_effectively_identical(self) -> bool:
        """The two splits are the same data: same samples, same counts."""
        return (
            self.n_train > 0
            and self.n_identical_rows == self.n_train == self.n_validation
        )


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)


def compare_splits(train: ad.AnnData, validation: ad.AnnData) -> SplitOverlap:
    """Measure what a training split and a validation split really have in common.

    Compares values, not just names, and counts the samples that match on both.
    """
    validation_names = set(validation.obs_names)
    shared = [name for name in train.obs_names if name in validation_names]

    n_identical = 0
    if shared:
        a = _dense(train[shared].X)
        b = _dense(validation[shared].X)
        if a.shape == b.shape:
            n_identical = int(np.all(a == b, axis=1).sum())

    return SplitOverlap(
        n_train=train.n_obs,
        n_validation=validation.n_obs,
        n_shared_names=len(shared),
        n_identical_rows=n_identical,
    )


def warn_if_validation_repeats_training(
    train_tissues: list[ad.AnnData],
    validation_tissues: list[ad.AnnData],
) -> list[SplitOverlap]:
    """Say so, loudly, when validation is really the training data again.

    Returns one :class:`SplitOverlap` per tissue so a caller can decide what to
    do. Warns rather than raises: a cohort too small to divide has nowhere else
    to go, and refusing would only push people back to the workaround. What it
    will not do is let the situation pass unremarked, which is what version 0.1
    did on every real dataset it was run on.
    """
    overlaps = [
        compare_splits(train, validation)
        for train, validation in zip(train_tissues, validation_tissues, strict=True)
    ]

    repeated = [i for i, o in enumerate(overlaps, start=1) if o.is_effectively_identical]
    if repeated:
        which = (
            f"tissue {repeated[0]}" if len(repeated) == 1 else f"tissues {repeated}"
        )
        warnings.warn(
            f"the validation data repeats the training data for {which}: the same "
            f"samples with the same counts. Any validation loss reported will be a "
            f"training loss, and will say nothing about how the model generalises. "
            f"If the cohort is too small to divide, leave validation_path unset "
            f"instead, and no validation loss will be reported.",
            ValidationRepeatsTrainingWarning,
            stacklevel=2,
        )

    # Keyed on samples that match by value, not by name. Keying it on names
    # reported the simple vignette, which is two independent draws that happen
    # to number their samples the same way, as fully overlapping.
    partial = [
        i
        for i, o in enumerate(overlaps, start=1)
        if not o.is_effectively_identical and o.n_identical_rows > 0
    ]
    if partial:
        detail = ", ".join(
            f"tissue {i}: {overlaps[i - 1].n_identical_rows} of "
            f"{overlaps[i - 1].n_train} ({overlaps[i - 1].fraction_repeated:.0%})"
            for i in partial
        )
        warnings.warn(
            f"some training samples reappear in the validation data ({detail}). "
            f"The validation loss will be optimistic by however much they overlap.",
            ValidationRepeatsTrainingWarning,
            stacklevel=2,
        )

    return overlaps
