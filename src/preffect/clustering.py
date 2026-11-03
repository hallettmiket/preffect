"""Group samples, and measure how well the grouping recovers known labels.

This is what the paper's subtype figures rest on, and what issues #4, #5 and
#16 need. Finding 30 established that a KL divergence cannot rank two models,
because it moves under a rescaling that leaves every prediction identical. A
clustering-recovery score does not move under that rescaling, which is why it
is the measure those issues now ask for.

Three things can be clustered, and version 0.1 had a separate function for each
with the resolution search copied into all three:

- the **latent space**, the model's summary of each sample
- the **cleaned counts**, what the model predicts each sample should look like
- the **observed counts**, as a baseline to beat

Here there is one function and a choice of input, because three copies of a
search is three places for it to go wrong, and in version 0.1 it had.

**Clustering needs extra packages.** ``pip install 'preffect[cluster]'``.
They are kept out of the core install because fitting a model does not need
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from preffect.errors import PreffectError

__all__ = [
    "ClusterResult",
    "RecoveryScores",
    "cluster_samples",
    "clustering_recovery",
    "resolution_for_target",
]

Source = Literal["latent", "cleaned_counts", "observed_counts"]

#: Where the resolution search gives up. Below this, Leiden returns essentially
#: one cluster per connected component and lowering it further does nothing.
MIN_RESOLUTION = 0.01


def _require_cluster_extras():
    """Import scanpy, or say what to install rather than raising ImportError."""
    try:
        import scanpy as sc
    except ImportError as error:  # pragma: no cover - exercised by hand
        raise PreffectError(
            "clustering needs scanpy, leidenalg and igraph, which are not "
            "installed. Run: pip install 'preffect[cluster]'"
        ) from error
    return sc


@dataclass(frozen=True)
class ClusterResult:
    """A grouping of samples, and the settings that produced it."""

    #: One integer per sample, in the order the samples were given.
    labels: np.ndarray
    #: The resolution the labels were actually produced at. See ``cluster_samples``.
    resolution: float
    n_clusters: int
    source: str
    #: True when the search stopped without reaching the requested count.
    hit_the_floor: bool = False
    #: Each resolution tried and how many clusters it gave, oldest first.
    search: tuple[tuple[float, int], ...] = field(default=())

    def __post_init__(self) -> None:
        if self.labels.ndim != 1:
            raise PreffectError(
                f"labels should be one per sample, got shape {self.labels.shape}"
            )

    def __len__(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class RecoveryScores:
    """How well a grouping recovers labels that were known in advance.

    Every score here is unchanged by relabelling the clusters, which matters
    because cluster 0 has no meaning: a grouping that is right but numbered
    differently must not be penalised.
    """

    adjusted_rand: float
    adjusted_mutual_information: float
    n_clusters: int
    n_true_groups: int
    n_samples: int

    def __str__(self) -> str:
        return (
            f"adjusted Rand {self.adjusted_rand:.3f}, "
            f"adjusted mutual information {self.adjusted_mutual_information:.3f} "
            f"({self.n_clusters} clusters against {self.n_true_groups} known groups, "
            f"{self.n_samples} samples)"
        )


def resolution_for_target(
    values: np.ndarray,
    *,
    target_clusters: int = 5,
    n_neighbors: int = 10,
    seed: int = 0,
    start: float = 1.0,
) -> ClusterResult:
    """Lower the Leiden resolution until the grouping has few enough clusters.

    Leiden does not take a number of clusters, it takes a resolution, and
    coarser groupings need lower numbers. Version 0.1 searched downward from 1.0
    for a resolution giving at most five clusters, which is the number of PAM50
    subtypes, and this keeps that behaviour with three corrections.

    **Version 0.1 reported a resolution its labels did not come from.** When the
    search bottomed out it did this::

        elif resolution <= 0.01:
            resolution = 0.1
            break

    It set the variable and broke *without re-running Leiden*, so the labels
    still came from a resolution at or below 0.01 while 0.1 was what got
    recorded. Here the search returns the resolution the labels were actually
    produced at, and says separately whether it ran out of room.

    **The steps are taken on a grid, not by repeated subtraction.** Version 0.1
    did ``resolution -= 0.1`` from 1.0, which reaches 0.30000000000000004 and
    then 0.20000000000000004, which is not ``<= 0.2``, so the branch meant to
    switch to finer steps fired one step later than intended and at a value no
    one would predict.

    **At most, not exactly.** The search only ever lowers the resolution, so it
    stops the moment the count is low enough and can return far fewer clusters
    than asked for, including one. That was true of version 0.1 too; what is
    new is that the result says so. ``target_clusters`` is a ceiling.

    :param values: samples by features.
    :param target_clusters: the most clusters wanted.
    :param n_neighbors: neighbours per sample when building the graph.
    :param seed: passed to Leiden, which is otherwise not repeatable.
    :param start: the first resolution tried.
    """
    import warnings

    import anndata as ad

    sc = _require_cluster_extras()

    if target_clusters < 1:
        raise PreffectError(f"target_clusters has to be at least 1, got {target_clusters}")
    values = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if values.ndim != 2:
        raise PreffectError(f"expected samples by features, got shape {values.shape}")
    n_samples = values.shape[0]
    if n_samples < 2:
        raise PreffectError(f"clustering needs at least 2 samples, got {n_samples}")

    adata = ad.AnnData(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Never ask for more neighbours than there are other samples.
        sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, n_samples - 1), use_rep="X")

        resolution = start
        search: list[tuple[float, int]] = []
        while True:
            sc.tl.leiden(
                adata,
                resolution=resolution,
                key_added="leiden",
                flavor="igraph",
                n_iterations=2,
                directed=False,
                random_state=seed,
            )
            labels = adata.obs["leiden"].to_numpy().astype(int)
            n_clusters = int(np.unique(labels).size)
            # The raw value, not a rounded copy of it. Rounding here made the
            # test that checks the steps land on a grid unable to fail: it read
            # back a number this line had already rounded. The grid is kept by
            # the step below, so an honest record shows it working.
            # Finding 37.
            search.append((resolution, n_clusters))

            if n_clusters <= target_clusters:
                hit_the_floor = False
                break
            if resolution <= MIN_RESOLUTION:
                # Out of room. Keep the labels this resolution produced rather
                # than reporting a resolution that was never run.
                hit_the_floor = True
                break
            # A grid, so every step lands on a value that can be written down.
            step = 0.01 if resolution <= 0.2 else 0.1
            resolution = round(max(MIN_RESOLUTION, resolution - step), 4)

    return ClusterResult(
        labels=labels,
        resolution=round(resolution, 4),
        n_clusters=n_clusters,
        source="values",
        hit_the_floor=hit_the_floor,
        search=tuple(search),
    )


def cluster_samples(
    result,
    source: Source = "latent",
    *,
    observed: np.ndarray | None = None,
    target_clusters: int = 5,
    n_neighbors: int = 10,
    seed: int = 0,
) -> ClusterResult:
    """Cluster what a fitted model produced, or the counts it started from.

    Replaces version 0.1's ``cluster_latent_space``, ``cluster_counts`` and
    ``cluster_true_counts``, which were three copies of one procedure differing
    only in the matrix they read.

    :param result: an ``InferenceResult`` from ``preffect.inference.infer``.
    :param source: ``latent``, ``cleaned_counts``, or ``observed_counts``.
    :param observed: required for ``observed_counts``, which is the baseline
        the model is supposed to beat and is not part of the model's output.
    """
    if source == "latent":
        values = result.latent.detach().cpu().numpy()
    elif source == "cleaned_counts":
        values = result.expected_counts.detach().cpu().numpy()
    elif source == "observed_counts":
        if observed is None:
            raise PreffectError(
                "clustering the observed counts needs them passed in as "
                "`observed`; they are the input to the model, not its output"
            )
        values = np.asarray(
            observed.detach().cpu().numpy() if hasattr(observed, "detach") else observed
        )
    else:
        raise PreffectError(
            f"source should be 'latent', 'cleaned_counts' or 'observed_counts', "
            f"got {source!r}"
        )

    found = resolution_for_target(
        values,
        target_clusters=target_clusters,
        n_neighbors=n_neighbors,
        seed=seed,
    )
    return ClusterResult(
        labels=found.labels,
        resolution=found.resolution,
        n_clusters=found.n_clusters,
        source=source,
        hit_the_floor=found.hit_the_floor,
        search=found.search,
    )


def clustering_recovery(labels, truth) -> RecoveryScores:
    """Score a grouping against labels that were known in advance.

    This is the functional measure issues #4, #5 and #16 ask for. Unlike a KL
    divergence it is unchanged by rescaling the latent space, so it compares
    two models rather than two gauges. See finding 30.

    Both scores are corrected for chance, which matters at these sizes: with
    five groups and a few hundred samples an uncorrected Rand index sits near
    0.7 for a random grouping and reads as success.

    :param labels: what the clustering produced, or a ``ClusterResult``.
    :param truth: the known group of each sample. Any labels; they are matched
        by identity, not by value, so strings are fine.
    """
    try:
        from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    except ImportError as error:  # pragma: no cover - exercised by hand
        raise PreffectError(
            "scoring a clustering needs scikit-learn, which is not installed. "
            "Run: pip install 'preffect[cluster]'"
        ) from error

    found = np.asarray(labels.labels if isinstance(labels, ClusterResult) else labels)
    known = np.asarray(truth)
    if found.shape[0] != known.shape[0]:
        raise PreffectError(
            f"there are {found.shape[0]} cluster labels and {known.shape[0]} "
            f"known labels; they have to match"
        )
    if found.shape[0] == 0:
        raise PreffectError("there is nothing to score")

    return RecoveryScores(
        adjusted_rand=float(adjusted_rand_score(known, found)),
        adjusted_mutual_information=float(
            adjusted_mutual_info_score(known, found)
        ),
        n_clusters=int(np.unique(found).size),
        n_true_groups=int(np.unique(known).size),
        n_samples=int(found.shape[0]),
    )
