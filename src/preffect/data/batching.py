"""Dividing samples into mini-batches.

A mini-batch is the handful of samples the model looks at before it adjusts
itself once. Smaller batches mean more adjustments per pass over the data.

**Every sample is used.** Version 0.1 skipped the final batch of each epoch
whenever it was smaller than the requested size, so with 250 samples and a
batch of 200, fifty never entered training and nothing said so. On one of the
published datasets that was 13 of 123 samples, every epoch. Here the last batch
is kept at whatever size it is, and dropping it is something you ask for.
That is defect 6 in ``how_this_project_breaks.md``.
"""

from __future__ import annotations

import numpy as np

from preffect.errors import ConfigError

__all__ = ["batch_indices", "n_batches"]


def n_batches(n_samples: int, batch_size: int, drop_last: bool = False) -> int:
    """How many batches one pass over the data will produce."""
    if n_samples <= 0:
        return 0
    whole, remainder = divmod(n_samples, batch_size)
    return whole if (drop_last or remainder == 0) else whole + 1


def batch_indices(
    n_samples: int,
    batch_size: int,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
    drop_last: bool = False,
) -> list[np.ndarray]:
    """Divide the sample positions into batches.

    :param rng: required when shuffling, so that a run with a seed is
        repeatable. Version 0.1 shuffled from the global random state, which is
        why its runs could not be reproduced even when a seed was set
        elsewhere.
    :param drop_last: discard a final short batch. Off by default.
    :returns: a list of arrays of positions. With ``drop_last`` off, every
        position from 0 to ``n_samples - 1`` appears exactly once across them.
    """
    if batch_size < 1:
        raise ConfigError(f"batch_size must be at least 1, got {batch_size}")
    if n_samples <= 0:
        return []

    if shuffle:
        if rng is None:
            raise ConfigError(
                "shuffling needs a random number generator, so that a run with a "
                "seed can be repeated. Pass one, or set shuffle=False."
            )
        order = rng.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    if batch_size > n_samples:
        # One batch of everything. Version 0.1 handled this by writing the
        # reduced size back into the shared settings, so the change outlived
        # the call and affected everything built afterwards.
        batch_size = n_samples

    batches = [order[i : i + batch_size] for i in range(0, n_samples, batch_size)]
    if drop_last and len(batches[-1]) < batch_size:
        batches.pop()
    return batches
