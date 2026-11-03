"""Hiding counts on purpose, so that recovering them can be scored.

Masking sets a fraction of the counts to zero before the model sees them. The
model is then asked to reconstruct the whole matrix, and the entries that were
hidden are the ones used to judge how well it imputes.

"Missing completely at random" (MCAR) means the hidden entries are chosen
without reference to their values or to anything about the sample: every entry
of a gene is equally likely to be hidden. It is the simplest of the standard
missingness patterns and the only one version 0.1 implemented, despite offering
two others in its settings that raised ``NotImplementedError`` when selected.

The fraction is applied **per gene**, matching version 0.1: each gene has the
same proportion of its samples hidden, rather than that proportion being spread
across the matrix as a whole. On a matrix with many all-zero genes the two
differ, so the behaviour is kept and written down rather than quietly improved.

Entries are hidden whether or not they were already zero. Nothing is scored as
the recovery of a zero by a zero: the relative error leaves zero-truth entries
out and reports how many (finding 25). What a hidden zero does is take nothing
away from the model's input, so "a fifth hidden" overstates what the model lost
by the zero share of the matrix: half a per cent at the project's gene
selection, a fifth of the entries on an unselected matrix. Measured in
``exp/14`` on 2026-10-29: drawing from the non-zero entries instead moves no
number outside noise, so the scheme stays and the count of entries an error
was computed over is reported beside it.

## Hiding entries lowers a sample's total, and the model is told by how much

A sample's library size is its total count, and it sets the scale of
everything the model predicts for that sample. Taken from a matrix with a
fifth of its entries zeroed it is a fifth too small, while the targets keep
the originals, so every predicted count came out short by the hidden fraction
and the error on the hidden entries was about 2.7 times what it should have
been (finding 56). :meth:`Mask.library_scale` is what repairs it: the factor
a row's observed total is multiplied by to estimate the total it would have
had, which is the number of genes divided by the number of entries of that row
the model was shown.

**The correction reads how many entries were hidden, never which values.**
The true total and the visible entries together give the sum of the hidden
ones exactly, so taking the library from the targets would hand the model the
answer it is being scored on. A count of hidden entries is protocol, known
before any value is looked at, and gives an estimate rather than the answer.

**It is available for "missing completely at random" and refused otherwise.**
The estimate is unbiased only because MCAR makes the entries the model was
shown a random subsample of the row, so their mean stands in for the row's.
Under a scheme that hides entries because of their values, the visible ones
are not representative and the same arithmetic would be quietly wrong. Both
other schemes raise ``NotImplementedError`` today, so the refusal costs
nothing now and could not be added later without auditing every result
produced in between.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preffect.errors import DataError

__all__ = ["MCAR", "Mask", "mcar_mask", "apply_mask"]


#: The only way of choosing hidden entries that is built, and the only one the
#: library-size correction is valid under. See the module docstring.
MCAR = "mcar"


@dataclass(frozen=True, slots=True)
class Mask:
    """Which entries were hidden, and how they were chosen.

    ``hidden`` is a boolean array the same shape as the counts, true where an
    entry was set to zero. ``scheme`` records how they were picked, because
    the library-size correction is only valid under one of them.
    """

    hidden: np.ndarray
    #: How the hidden entries were chosen. ``"mcar"``, missing completely at
    #: random, is the only scheme built, and the only one
    #: :meth:`library_scale` will answer for.
    scheme: str = MCAR

    @property
    def n_hidden(self) -> int:
        return int(self.hidden.sum())

    @property
    def fraction_hidden(self) -> float:
        return self.n_hidden / self.hidden.size if self.hidden.size else 0.0

    def positions(self) -> np.ndarray:
        """The hidden entries as an array of (sample, gene) pairs."""
        return np.argwhere(self.hidden)

    def visible_per_sample(self) -> np.ndarray:
        """How many of each sample's genes the model was shown, as ``[samples]``."""
        return self.hidden.shape[1] - self.hidden.sum(axis=1)

    def library_scale(self) -> np.ndarray:
        """What a sample's observed total is multiplied by to estimate its true total.

        The number of genes divided by the number of that sample's entries the
        model was shown. A sample with a fifth of its entries hidden has its
        observed total raised by about a quarter, and the exact factor differs
        from sample to sample because the hidden entries are drawn per gene
        rather than per sample.

        Returned as ``[samples, 1]`` so it multiplies a column of row totals
        directly.

        :raises DataError: for any scheme but ``"mcar"``, naming why. The
            estimate stands on the shown entries being a random subsample of
            the sample's, which only missing completely at random guarantees.
        :raises DataError: if any sample has no visible entry at all, since
            nothing is then left to estimate its total from.
        """
        if self.scheme != MCAR:
            raise DataError(
                f"the library-size correction under a mask is only valid for the "
                f"{MCAR!r} scheme and this mask says {self.scheme!r}. The "
                f"correction multiplies a sample's observed total by its genes "
                f"over the entries it was shown, which estimates the true total "
                f"only when the shown entries are a random subsample of the "
                f"sample's. A scheme that hides entries according to their "
                f"values breaks that, and the estimate would be biased rather "
                f"than unavailable."
            )
        n_genes = self.hidden.shape[1]
        visible = self.visible_per_sample()
        if (visible == 0).any():
            empty = int((visible == 0).sum())
            raise DataError(
                f"{empty} sample(s) have every entry hidden, so there is nothing "
                f"left to estimate their total counts from. Hide a smaller "
                f"fraction."
            )
        return (n_genes / visible).astype(np.float32).reshape(-1, 1)


def mcar_mask(
    n_samples: int,
    n_genes: int,
    fraction: float,
    rng: np.random.Generator,
) -> Mask:
    """Choose which entries to hide, the same proportion within every gene.

    :param fraction: the proportion of each gene's samples to hide, between 0
        and 1. The number hidden per gene is rounded down, so a fraction below
        ``1 / n_samples`` hides nothing and says so.
    """
    if not 0.0 <= fraction <= 1.0:
        raise DataError(f"fraction must lie between 0 and 1, got {fraction}")

    hidden = np.zeros((n_samples, n_genes), dtype=bool)
    per_gene = int(fraction * n_samples)
    if per_gene == 0:
        if fraction > 0:
            raise DataError(
                f"a fraction of {fraction} over {n_samples} samples rounds down to "
                f"no hidden entries per gene, so nothing would be masked. Use a "
                f"fraction of at least {1 / n_samples:.3g}, or turn masking off."
            )
        return Mask(hidden=hidden, scheme=MCAR)

    # One independent draw per gene, vectorised. Version 0.1 looped in Python
    # over every gene and called random.sample inside, which on a
    # twenty-thousand gene matrix is twenty thousand calls.
    for gene in range(n_genes):
        chosen = rng.choice(n_samples, size=per_gene, replace=False)
        hidden[chosen, gene] = True

    return Mask(hidden=hidden, scheme=MCAR)


def apply_mask(counts: np.ndarray, mask: Mask) -> np.ndarray:
    """Return a copy of the counts with the hidden entries set to zero."""
    if counts.shape != mask.hidden.shape:
        raise DataError(
            f"the mask is shaped {mask.hidden.shape} but the counts are "
            f"{counts.shape}; they have to match"
        )
    masked = counts.copy()
    masked[mask.hidden] = 0
    return masked
