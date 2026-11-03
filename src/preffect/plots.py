"""Figures for looking at what a fitted model did.

These are diagnostics, not publication figures: they are for deciding whether a
run worked. Every one returns a matplotlib ``Figure`` and **none of them write
a file or open a window**. Saving is a separate step, so a caller chooses where
things land rather than inheriting one machine's paths. Version 0.1 shipped
absolute paths as defaults and they reached a published figure.

Three things are deliberately different from version 0.1's six ``visualize_*``
methods:

**Genes are named, not counted.** Version 0.1's scatter plot drew
``for x in range(0, 50)``, which is column *positions* 0 to 49. Gene order in
the matrix is whatever the file or the gene selection produced, so the figure
showed an arbitrary subset that changed when the order did, and it raised
``IndexError`` below 50 genes. Here you pass gene names, and the default is the
most variable genes, which is a stated rule rather than an accident of layout.

**The UMAP is seeded.** Version 0.1 built ``umap.UMAP(metric="cosine",
n_neighbors=30)`` with no ``random_state`` in the one figure that goes in a
paper, while seeding other UMAP calls in the same tree. UMAP is stochastic, so
the same model and data gave a different picture every run.

**The figures do not reach into a model.** They take what inference returned
plus the counts, so a figure can be drawn from a saved result without rebuilding
anything.

Needs ``pip install 'preffect[viz]'``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from preffect.errors import PreffectError

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

__all__ = [
    "save_figure",
    "library_size_histogram",
    "gene_scatter",
    "gene_dispersion_and_zeros",
    "sample_fit_spread",
    "batch_adjustment",
    "zero_fraction_against_pi",
    "library_size_and_dispersion",
    "latent_umap",
    "loss_curves",
    "subtype_heatmap",
    "SUBTYPE_ORDER",
]

#: Version 0.1 compared every pair of batches, which grows as the square, and
#: capped the figure at the first five. Kept, because the cap is sensible and
#: the alternative is a figure nobody can read.
MAX_BATCHES = 5


def _figure(n_rows: int = 1, n_columns: int = 1, size=(5.0, 4.0)):
    """A figure with no global state and no backend, so it works headless.

    Uses ``Figure`` directly rather than ``pyplot``, which keeps a registry of
    open windows and leaks them in a loop.
    """
    try:
        from matplotlib.figure import Figure
    except ImportError as error:  # pragma: no cover - exercised by hand
        raise PreffectError(
            "drawing needs matplotlib, which is not installed. "
            "Run: pip install 'preffect[viz]'"
        ) from error

    figure = Figure(figsize=(size[0] * n_columns, size[1] * n_rows), layout="constrained")
    axes = figure.subplots(n_rows, n_columns, squeeze=False)
    return figure, axes


def _as_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values)


def save_figure(figure: Figure, path) -> None:
    """Write a figure, creating the folder if it is missing.

    Separate from drawing on purpose. A function that draws *and* saves cannot
    be used to build a panel of several, and has to be told a path it should
    not have an opinion about.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def library_size_histogram(result, observed) -> Figure:
    """Observed library sizes against what the model inferred.

    The library size is the total counts in a sample. If the model's inferred
    sizes sit on top of the observed ones, it has learned the depth of
    sequencing; if they are shifted or much narrower, it has not.
    """
    figure, axes = _figure(1, 2)
    observed_totals = _as_array(observed).sum(axis=1)
    inferred_totals = _as_array(result.library_size).ravel()

    for axis, values, title, colour in (
        (axes[0][0], observed_totals, "Observed", "tab:blue"),
        (axes[0][1], inferred_totals, "Inferred by the model", "tab:orange"),
    ):
        axis.hist(values, bins=30, color=colour, edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel("total counts in a sample")
        axis.set_ylabel("samples")

    # Same limits on both, or two different pictures look like the same one.
    both = np.concatenate([observed_totals, inferred_totals])
    for axis in (axes[0][0], axes[0][1]):
        axis.set_xlim(both.min(), both.max())
    return figure


def _most_variable(values: np.ndarray, n: int) -> np.ndarray:
    """Column indices of the ``n`` most variable genes, largest first."""
    spread = values.std(axis=0)
    return np.argsort(spread)[::-1][:n]


def gene_scatter(
    result,
    observed,
    *,
    gene_names: Sequence[str] | None = None,
    genes: Sequence[str] | None = None,
    n_genes: int = 12,
) -> Figure:
    """Observed against fitted counts, one panel per gene.

    A gene whose points lie on the diagonal is one the model reproduces.

    :param genes: which genes to draw, **by name**. Defaults to the most
        variable, because those are the ones a model can be wrong about. Version
        0.1 drew columns 0 to 49, which is a different set of genes every time
        the gene order changes.
    :param gene_names: names for the columns, needed to select by name.
    """
    observed_values = _as_array(observed)
    fitted = _as_array(result.expected_counts)
    if observed_values.shape != fitted.shape:
        raise PreffectError(
            f"observed counts are {observed_values.shape} and fitted are "
            f"{fitted.shape}; they have to match"
        )

    n_available = observed_values.shape[1]
    if genes is not None:
        if gene_names is None:
            raise PreffectError(
                "selecting genes by name needs `gene_names` for the columns"
            )
        lookup = {str(name): i for i, name in enumerate(gene_names)}
        missing = [g for g in genes if str(g) not in lookup]
        if missing:
            raise PreffectError(
                f"these genes are not in the data: {', '.join(map(str, missing[:5]))}"
            )
        chosen = np.array([lookup[str(g)] for g in genes])
    else:
        chosen = _most_variable(observed_values, min(n_genes, n_available))

    columns = min(4, len(chosen))
    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = _figure(rows, columns, size=(3.0, 3.0))

    for position, column in enumerate(chosen):
        axis = axes[position // columns][position % columns]
        axis.scatter(observed_values[:, column], fitted[:, column], s=8, alpha=0.6)
        limit = max(observed_values[:, column].max(), fitted[:, column].max(), 1.0)
        axis.plot([0, limit], [0, limit], linestyle="--", linewidth=1, color="grey")
        label = str(gene_names[column]) if gene_names is not None else f"column {column}"
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("observed")
        axis.set_ylabel("fitted")

    for empty in range(len(chosen), rows * columns):  # blank any unused panels
        axes[empty // columns][empty % columns].axis("off")
    return figure


def batch_adjustment(result, observed, batches) -> Figure:
    """Per-gene means of one batch against another, observed and fitted.

    If batch correction worked, the fitted panel is closer to the diagonal than
    the observed one: the model has removed a difference that was in the data.
    Every pair of batches is drawn, capped at the first ``MAX_BATCHES`` because
    the number of pairs grows as the square.
    """
    observed_values = _as_array(observed)
    fitted = _as_array(result.expected_counts)
    labels = _as_array(batches).ravel()
    if labels.shape[0] != observed_values.shape[0]:
        raise PreffectError(
            f"there are {labels.shape[0]} batch labels and "
            f"{observed_values.shape[0]} samples; they have to match"
        )

    present = list(dict.fromkeys(labels.tolist()))[:MAX_BATCHES]
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1 :]]
    if not pairs:
        raise PreffectError(
            f"batch comparison needs at least two batches, found {len(present)}"
        )

    figure, axes = _figure(len(pairs), 2, size=(4.0, 3.6))
    for row, (first, second) in enumerate(pairs):
        for column, (values, title) in enumerate(
            ((observed_values, "Observed"), (fitted, "Fitted"))
        ):
            axis = axes[row][column]
            x = values[labels == first].mean(axis=0)
            y = values[labels == second].mean(axis=0)
            axis.scatter(x, y, s=6, alpha=0.5)
            limit = max(x.max(), y.max(), 1.0)
            axis.plot([0, limit], [0, limit], linestyle="--", linewidth=1, color="grey")
            axis.set_title(f"{title}: batch {first} against {second}", fontsize=9)
            axis.set_xlabel(f"mean count, batch {first}")
            axis.set_ylabel(f"mean count, batch {second}")
    return figure


def zero_fraction_against_pi(result, observed) -> Figure:
    """How often a gene is zero, against the model's own zero probability.

    Only meaningful for the zero-inflated likelihood, which is the one with a
    separate probability of a structural zero. Refused otherwise rather than
    drawn empty: version 0.1's habit of quietly producing something was the
    most expensive thing in it.
    """
    if result.zero_logits is None:
        raise PreffectError(
            "this figure needs the zero-inflated likelihood (ZINB); the fitted "
            "model has no zero probability to plot. Refit with likelihood=ZINB."
        )

    observed_values = _as_array(observed)
    probability = 1.0 / (1.0 + np.exp(-_as_array(result.zero_logits)))

    figure, axes = _figure()
    axis = axes[0][0]
    axis.scatter(
        (observed_values == 0).mean(axis=0), probability.mean(axis=0), s=8, alpha=0.6
    )
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
    axis.set_xlabel("fraction of samples where the gene is zero")
    axis.set_ylabel("mean fitted probability of a zero")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    return figure


def library_size_and_dispersion(result, observed) -> Figure:
    """Four panels: gene means, dispersion, library size, and mean against dispersion.

    The last panel is the one worth looking at. In a negative binomial, the
    dispersion sets how much more variable the counts are than a Poisson would
    be, and it usually falls as the mean rises. A flat or rising relationship
    suggests the dispersion is not being fitted.
    """
    observed_values = _as_array(observed)
    fitted = _as_array(result.expected_counts)
    dispersion = _as_array(result.dispersion)

    figure, axes = _figure(2, 2, size=(4.5, 3.6))

    axes[0][0].scatter(
        observed_values.mean(axis=0), fitted.mean(axis=0), s=6, alpha=0.5
    )
    limit = max(observed_values.mean(axis=0).max(), fitted.mean(axis=0).max(), 1.0)
    axes[0][0].plot([0, limit], [0, limit], "--", linewidth=1, color="grey")
    axes[0][0].set_title("Gene means", fontsize=10)
    axes[0][0].set_xlabel("observed")
    axes[0][0].set_ylabel("fitted")

    axes[0][1].hist(dispersion.mean(axis=0), bins=30, color="tab:green", edgecolor="white")
    axes[0][1].set_title("Fitted dispersion per gene", fontsize=10)
    axes[0][1].set_xlabel("dispersion")
    axes[0][1].set_ylabel("genes")

    axes[1][0].scatter(
        observed_values.sum(axis=1), _as_array(result.library_size).ravel(), s=8, alpha=0.6
    )
    axes[1][0].set_title("Library size", fontsize=10)
    axes[1][0].set_xlabel("observed total")
    axes[1][0].set_ylabel("fitted total")

    mean_counts = np.clip(observed_values.mean(axis=0), 1e-8, None)
    axes[1][1].scatter(mean_counts, dispersion.mean(axis=0), s=6, alpha=0.5)
    axes[1][1].set_xscale("log")
    axes[1][1].set_yscale("log")
    axes[1][1].set_title("Mean against dispersion", fontsize=10)
    axes[1][1].set_xlabel("mean count (log scale)")
    axes[1][1].set_ylabel("dispersion (log scale)")
    return figure


def gene_dispersion_and_zeros(result, observed, *, gene_names, gene) -> Figure:
    """One gene's fitted dispersion across the samples, and its chance of a zero.

    Both quantities are fitted for every sample separately, so for a single
    gene each is a spread rather than a number, and the spread is the thing
    worth seeing: a dispersion that is the same in every sample says the model
    found the gene equally erratic throughout.

    The left panel is the fitted dispersion, which is how much more variable
    the counts are than pure counting noise would make them. The right panel,
    drawn only for the zero-inflated model, is the fitted probability that
    this gene's count is a structural zero, with a dashed line at the share of
    samples in which it was actually observed to be zero. The two should agree.

    :param gene_names: names for the columns, in order.
    :param gene: which gene to draw, by name.
    """
    observed_values = _as_array(observed)
    column = _gene_index(gene_names, gene)
    dispersion = _as_array(result.dispersion)[:, column]
    has_zeros = result.zero_logits is not None

    figure, axes = _figure(1, 2 if has_zeros else 1, size=(4.5, 3.6))
    axes[0][0].hist(dispersion, bins=20, color="tab:green", edgecolor="white")
    axes[0][0].set_title(f"{gene}: fitted dispersion", fontsize=10)
    axes[0][0].set_xlabel("dispersion")
    axes[0][0].set_ylabel("samples")

    if has_zeros:
        probability = 1.0 / (1.0 + np.exp(-_as_array(result.zero_logits)[:, column]))
        axes[0][1].hist(probability, bins=20, color="tab:purple", edgecolor="white")
        axes[0][1].axvline(
            float((observed_values[:, column] == 0).mean()),
            linestyle="--",
            linewidth=1,
            color="grey",
        )
        axes[0][1].set_title(f"{gene}: fitted chance of a zero", fontsize=10)
        axes[0][1].set_xlabel("probability of a structural zero")
        axes[0][1].set_ylabel("samples")
        axes[0][1].set_xlim(0, 1)
    return figure


def sample_fit_spread(
    scores,
    *,
    highlight: int | None = None,
    measure: str = "mean log probability per gene",
) -> Figure:
    """How one number per sample is spread over the cohort, with one sample marked.

    Takes the scores rather than computing them, so that a table of the same
    numbers and this figure cannot disagree about what they are. The dashed
    line is the sample asked about, and what it is for is placing that sample
    against the rest: a tail far to the left is the handful of samples the
    model cannot account for.

    :param scores: one value per sample.
    :param highlight: the position of one sample to mark, or ``None``.
    :param measure: what the scores are, for the axis label.
    """
    values = _as_array(scores).ravel()
    if values.size == 0:
        raise PreffectError("there are no sample scores to draw")

    figure, axes = _figure(1, 1, size=(6.0, 3.6))
    axes[0][0].hist(values, bins=min(30, max(5, values.size // 2)),
                    color="tab:blue", edgecolor="white")
    if highlight is not None:
        axes[0][0].axvline(
            float(values[highlight]), linestyle="--", linewidth=1.5, color="tab:red"
        )
    axes[0][0].set_xlabel(measure)
    axes[0][0].set_ylabel("samples")
    return figure


def _gene_index(gene_names, gene) -> int:
    """Where a named gene sits among the columns, refused by name if absent."""
    if gene_names is None:
        raise PreffectError("selecting a gene by name needs `gene_names` for the columns")
    lookup = {str(name): i for i, name in enumerate(gene_names)}
    if str(gene) not in lookup:
        raise PreffectError(f"{gene} is not one of the genes in the data")
    return lookup[str(gene)]


def latent_umap(
    result,
    *,
    colour_by=None,
    colour_label: str = "group",
    seed: int = 0,
    n_neighbors: int = 30,
    metric: str = "cosine",
) -> Figure:
    """The latent space in two dimensions, optionally coloured by a known label.

    This is the figure the paper's subtype panels are built from, and it is the
    one version 0.1 left unseeded, so the same model and data drew a different
    picture every run. ``seed`` fixes that; pass ``seed=None`` deliberately if
    you want the old behaviour.

    **Seeding costs parallelism, and UMAP says so**: it drops to one thread
    when given a ``random_state``, because its parallel path is not
    deterministic. On a vignette this is invisible. On 528 samples it is worth
    knowing before you wonder why the figure is slow, and it is the trade this
    function makes on purpose: a figure you cannot reproduce is not worth the
    time it saved.

    Replaces both ``visualize_latent_recons_umap`` and
    ``draw_umap_with_subtype``: they differed only in what coloured the points.

    :param colour_by: one label per sample. Subtypes, batches, anything.
    """
    try:
        import umap
    except ImportError as error:  # pragma: no cover - exercised by hand
        raise PreffectError(
            "the UMAP figure needs umap-learn, which is not installed. "
            "Run: pip install 'preffect[viz]'"
        ) from error

    latent = _as_array(result.latent)
    n_samples = latent.shape[0]
    if n_samples < 3:
        raise PreffectError(f"a UMAP needs at least 3 samples, got {n_samples}")

    reducer = umap.UMAP(
        metric=metric,
        # Never ask for more neighbours than there are other samples. Version
        # 0.1 hardcoded 30 and failed on any cohort smaller than that.
        n_neighbors=min(n_neighbors, n_samples - 1),
        random_state=seed,
    )
    embedding = reducer.fit_transform(latent)

    figure, axes = _figure(size=(5.5, 4.5))
    axis = axes[0][0]
    if colour_by is None:
        axis.scatter(embedding[:, 0], embedding[:, 1], s=14, alpha=0.8)
    else:
        labels = _as_array(colour_by).ravel()
        if labels.shape[0] != n_samples:
            raise PreffectError(
                f"there are {labels.shape[0]} labels and {n_samples} samples; "
                f"they have to match"
            )
        for value in dict.fromkeys(labels.tolist()):
            rows = labels == value
            axis.scatter(
                embedding[rows, 0], embedding[rows, 1], s=14, alpha=0.8, label=str(value)
            )
        axis.legend(title=colour_label, fontsize=8, title_fontsize=9)

    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    # Said on the figure, because a UMAP's axes have no units and the distances
    # between well-separated clumps do not mean anything.
    axis.set_title(
        f"Latent space, UMAP (seed {seed}, {metric} distance)", fontsize=10
    )
    return figure


def loss_curves(history) -> Figure:
    """The objective over epochs, in total and term by term.

    Replaces ``plot_progression`` and ``plot_progression_all``. Validation is
    drawn on the same axes as training where it exists, because the gap between
    them is the thing worth seeing and two separate figures hide it.
    """
    epochs = [record.epoch for record in history.epochs]
    if not epochs:
        raise PreffectError("there are no epochs to draw")

    term_names = list(history.epochs[0].train.mean_terms)
    figure, axes = _figure(1, 2, size=(5.5, 4.0))

    axes[0][0].plot(epochs, history.train_losses, label="training", linewidth=2)
    validation = [
        record.validation.mean_loss if record.validation else None
        for record in history.epochs
    ]
    if any(v is not None for v in validation):
        drawn = [(e, v) for e, v in zip(epochs, validation, strict=True) if v is not None]
        axes[0][0].plot(
            [e for e, _ in drawn], [v for _, v in drawn], label="validation", linewidth=2
        )
        axes[0][0].legend(fontsize=9)
    axes[0][0].set_title("Objective", fontsize=10)
    axes[0][0].set_xlabel("epoch")
    axes[0][0].set_ylabel("mean loss per batch")

    for name in term_names:
        axes[0][1].plot(
            epochs,
            [record.train.mean_terms[name] for record in history.epochs],
            label=name,
            linewidth=1.5,
        )
    axes[0][1].set_yscale("symlog")
    axes[0][1].set_title("Each term (symmetric log scale)", fontsize=10)
    axes[0][1].set_xlabel("epoch")
    axes[0][1].set_ylabel("mean value per batch")
    axes[0][1].legend(fontsize=8)

    if history.stopped_early and history.best_epoch is not None:
        for axis in (axes[0][0], axes[0][1]):
            axis.axvline(
                history.best_epoch, linestyle=":", color="grey", linewidth=1
            )
    return figure


#: How subtypes are ordered along the heatmap unless told otherwise. Roughly
#: least to most aggressive, which is how they are read. Version 0.1 ordered
#: them by the **alphabetical order of the colour names** it had assigned, so
#: changing a colour changed the layout of the figure. See finding 33.
SUBTYPE_ORDER = ("Normal", "LumA", "LumB", "Her2", "Basal", "Claudin")


def _ordered_groups(labels: np.ndarray, order: Sequence[str] | None) -> list:
    """Known subtypes in a stated order, then anything else, alphabetically."""
    present = list(dict.fromkeys(labels.tolist()))
    if order is None:
        order = SUBTYPE_ORDER
    known = [name for name in order if name in present]
    rest = sorted(str(name) for name in present if name not in set(order))
    return known + rest


def subtype_heatmap(
    values,
    gene_names: Sequence[str],
    subtypes,
    *,
    gene_set=None,
    normalise: str = "z_score",
    subtype_order: Sequence[str] | None = None,
    min_genes: int = 10,
    title: str = "PAM50 genes by subtype",
) -> Figure:
    """Expression of the PAM50 genes, with samples grouped by subtype.

    This is the figure the paper's subtype panel is built from. Rows are
    samples, grouped and ordered by subtype with a colour bar on the left;
    columns are the genes of the set that are present in the data.

    Four things differ from version 0.1's ``create_PAM50_heatmap``, all of them
    finding 33.

    **The genes are found whichever way the data names them.** Version 0.1
    matched against a list of symbols, and three of the nine datasets name their
    genes with Ensembl identifiers. It matched nothing on those, printed
    ``PAM50 Genes Found: 0``, and drew a heatmap of an empty matrix. Here both
    namings are tried and the better match is used.

    **Too few matches is refused, not drawn.** ``min_genes`` is the floor.

    **Any number of subtypes works.** Version 0.1 indexed a hardcoded list of
    five colours by subtype, so a dataset with six categories raised
    ``IndexError``. Two of the nine have six: ``our_dcis`` has an explicit
    ``NaN`` group and GSE209998 has ``Claudin``.

    **Samples are ordered by subtype, not by colour name.** Version 0.1 sorted
    on the colour string it had assigned, so the groups came out in the
    alphabetical order of five colour names and changing a colour changed the
    figure.

    :param values: samples by genes. Counts, or a model's cleaned counts.
    :param normalise: ``z_score`` per gene, ``log_z_score`` for counts spanning
        orders of magnitude, or ``min_max`` to match version 0.1. Min-max is
        set by the single largest value in a gene, so one outlier flattens the
        rest; it is here for comparison, not because it is a good default.
    """
    from preffect.gene_sets import PAM50, match_genes

    gene_set = gene_set or PAM50
    matrix = np.nan_to_num(_as_array(values).astype(float))
    labels = _as_array(subtypes).ravel()

    if matrix.shape[0] != labels.shape[0]:
        raise PreffectError(
            f"there are {matrix.shape[0]} samples and {labels.shape[0]} subtype "
            f"labels; they have to match"
        )
    if matrix.shape[1] != len(gene_names):
        raise PreffectError(
            f"there are {matrix.shape[1]} columns and {len(gene_names)} gene "
            f"names; they have to match"
        )

    positions, found, naming = match_genes(gene_set, gene_names)
    if len(positions) < min_genes:
        raise PreffectError(
            f"only {len(positions)} of {len(gene_set)} {gene_set.name} genes are "
            f"in this data, which is below min_genes={min_genes}. Tried both "
            f"gene symbols and Ensembl identifiers. Check that the gene names "
            f"are what you think they are: this data names its first gene "
            f"{str(gene_names[0])!r}."
        )

    matrix = matrix[:, positions]
    if normalise == "z_score":
        spread = matrix.std(axis=0)
        scaled = (matrix - matrix.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    elif normalise == "log_z_score":
        logged = np.log1p(np.clip(matrix, 0, None))
        spread = logged.std(axis=0)
        scaled = (logged - logged.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    elif normalise == "min_max":
        low, high = matrix.min(axis=0), matrix.max(axis=0)
        scaled = (matrix - low) / np.where(high > low, high - low, 1.0)
    else:
        raise PreffectError(
            f"normalise should be 'z_score', 'log_z_score' or 'min_max', "
            f"got {normalise!r}"
        )

    groups = _ordered_groups(labels, subtype_order)
    row_order = np.concatenate([np.flatnonzero(labels == group) for group in groups])
    scaled = scaled[row_order]

    import matplotlib as mpl

    # A colour per subtype from a colormap, so any number of them works.
    palette = mpl.colormaps["tab10"].resampled(max(len(groups), 1))
    colour_of = {group: palette(i) for i, group in enumerate(groups)}

    from matplotlib.figure import Figure

    # Built by hand rather than through _figure, because the subtype bar is a
    # thin strip beside the heatmap, not a panel of equal width.
    figure = Figure(
        figsize=(max(8.0, 0.22 * len(found) + 4.0), 7.0), layout="constrained"
    )
    bar, main = figure.subplots(
        1, 2, gridspec_kw={"width_ratios": [1, 28], "wspace": 0.02}
    )

    bar.imshow(
        np.array([[colour_of[group] for group in labels[row_order]]]).transpose(1, 0, 2),
        aspect="auto",
        interpolation="nearest",
    )
    bar.set_xticks([])
    bar.set_yticks([])
    bar.set_ylabel(f"{len(row_order)} samples, grouped by subtype")

    limit = float(np.abs(scaled).max()) if normalise != "min_max" else 1.0
    image = main.imshow(
        scaled,
        aspect="auto",
        cmap="RdBu_r" if normalise != "min_max" else "hot",
        vmin=-limit if normalise != "min_max" else 0.0,
        vmax=limit,
        interpolation="nearest",
    )
    main.set_xticks(range(len(found)))
    main.set_xticklabels(found, rotation=90, fontsize=6)
    main.set_yticks([])
    main.set_title(f"{title} ({len(found)} genes matched by {naming}, {normalise})",
                   fontsize=10)

    figure.colorbar(image, ax=main, shrink=0.6, label=normalise.replace("_", " "))
    main.legend(
        handles=[
            mpl.patches.Patch(
                color=colour_of[group],
                label=f"{group} ({int((labels == group).sum())})",
            )
            for group in groups
        ],
        loc="center left",
        bbox_to_anchor=(1.18, 0.5),
        fontsize=8,
        title="subtype",
        title_fontsize=9,
    )
    return figure
