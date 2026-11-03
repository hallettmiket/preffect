"""Turning a counts table into the file PREFFECT reads: network, correspondence table, writer.

This module is for someone who has a table of counts and a table describing
the samples, and needs the ``.h5ad`` file the rest of the package reads.
Preparing that file is the step people stumble on, and until now nothing in
the package helped with it: the layout was described in the documentation and
left to be assembled by hand.

Every function here takes plain arrays and data frames rather than paths, so
the browser interface can call them with what somebody uploaded and the
command line can call them with what it read off disk.

## What a tissue file has to carry

A tissue is one sample from each patient under one condition, for example a
tumour or the normal tissue beside it. One file holds one tissue of one split,
and its name has to end in ``tau_<n>.h5ad``, where ``<n>`` is the tissue
number counting from 1, because that is what the loader looks for.

Inside the file:

- the counts, samples down the rows and genes across the columns, as whole
  numbers that are not negative;
- unique gene names, which are the key everything else matches on;
- whatever the study knows about each sample, such as a subtype or a batch;
- optionally the sample-sample network under ``obsm['sample_sample_adj']``,
  which says which samples resemble each other and is the only thing the
  single and full models attend over;
- optionally the correspondence table under ``obsm['sample_association']``,
  which says which row of one tissue is the same patient as which row of
  another, and which the full model cannot run without.

## Where the network recipe comes from

:func:`nearest_neighbour_network` builds the network this project measured
rather than the one version 0.1 described. No dataset version 0.1 shipped
carries a network at all: each stores the identity matrix, which links every
sample to itself and to nothing else, because the preparation script's
thresholding step modified a copy and returned nothing and its cast to a
small integer type then kept only correlations that round to exactly one
(findings 35 and 54). The recipe here is ``exp/16_cohort_sample_network``'s
and the decision of 2026-10-30 on the cohort's sample network: square-root
the counts, take the Pearson correlation between every pair of samples, link
each sample to the ten others it correlates with most, and keep an edge
wherever either of the two names the other.

Ten neighbours is the default because it is the one candidate under which the
graph model beat the simple model on the project's own cohort: imputation
error at hidden entries 8 per cent lower in six seeds of six. The threshold
the old recipe meant to apply, 0.8, links 57 per cent of all pairs of that
cohort and makes the graph model worse than the simple one.

## What these functions refuse

They refuse rather than repair, because a count matrix quietly coerced into
shape is how version 0.1's problems reached results. The refusals are: counts
that are not whole and not negative, naming the entries at fault; a repeated
sample name or gene name; a number of neighbours that leaves the network with
no edges or with every edge; a patient listed twice in one tissue; and
writing over a file that is already there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from preffect.errors import DataError

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from collections.abc import Sequence

    import anndata as ad
    import pandas as pd

__all__ = [
    "DEFAULT_NEIGHBOURS",
    "MISSING_COUNTERPART",
    "NetworkSummary",
    "WrittenNetwork",
    "add_network_to_file",
    "correspondence_table",
    "describe_sample_network",
    "nearest_neighbour_network",
    "read_counts_table",
    "read_sample_table",
    "sample_correlations",
    "tissue_file_name",
    "write_tissue",
]

#: How many neighbours each sample gets by default, from the decision of
#: 2026-10-30 on the cohort's sample network.
DEFAULT_NEIGHBOURS = 10

#: What a correspondence table holds where a patient has no sample in a
#: tissue. Written as this string rather than as a missing value because
#: pandas 2 makes a column of names with a missing value in it an object
#: column that anndata refuses to write, and because version 0.1 wrote it
#: this way and the reader turns it back into a missing value.
MISSING_COUNTERPART = "NA"


# ---------------------------------------------------------------------------
# Building a network from counts
# ---------------------------------------------------------------------------


def sample_correlations(counts: Any) -> np.ndarray:
    """How alike every pair of samples is, on square-rooted counts.

    The similarity the network is built from: genes with no count anywhere are
    dropped, the rest are square-rooted, and the Pearson correlation is taken
    between every pair of samples. Square-rooting is what version 0.1's
    preparation did, and it keeps a handful of very highly expressed genes
    from deciding the answer on their own.

    :param counts: samples down the rows, genes across the columns. An array
        or a data frame; a data frame's values are used.
    :returns: a samples by samples matrix of correlations, with one on the
        diagonal.
    :raises DataError: if there are fewer than two samples, if any count is
        missing, or if a sample's counts are all the same value, which leaves
        its correlation with everything undefined.
    """
    values = _as_array(counts, "counts")
    if values.ndim != 2:
        raise DataError(
            f"counts have {values.ndim} dimension(s); they must be a table with "
            f"samples down the rows and genes across the columns"
        )
    n_samples = values.shape[0]
    if n_samples < 2:
        raise DataError(
            f"a sample-sample network needs at least two samples, and there "
            f"{'is' if n_samples == 1 else 'are'} {n_samples}"
        )

    if np.isnan(values).any():
        raise DataError(
            f"{int(np.isnan(values).sum())} count(s) are missing, and a "
            f"correlation over them is undefined. Set them to 0, or drop the "
            f"gene or sample, before building a network."
        )

    kept = values.any(axis=0)
    if not kept.any():
        raise DataError("every count is zero, so no two samples can be told apart")

    rooted = np.sqrt(values[:, kept].astype(np.float64))
    flat = np.flatnonzero(rooted.std(axis=1) == 0)
    if flat.size:
        raise DataError(
            f"{flat.size} sample(s) hold the same value at every gene, at row(s) "
            f"{flat[:5].tolist()}{' and more' if flat.size > 5 else ''}, so their "
            f"correlation with every other sample is undefined. Drop them before "
            f"building a network."
        )
    return np.corrcoef(rooted)


def nearest_neighbour_network(
    counts: Any,
    neighbours: int = DEFAULT_NEIGHBOURS,
    *,
    mutual: bool = False,
    correlation: np.ndarray | None = None,
) -> np.ndarray:
    """Link each sample to the samples whose counts look most like its own.

    The recipe ``exp/16_cohort_sample_network`` measured and the decision of
    2026-10-30 adopted: the correlation of :func:`sample_correlations`, then
    each sample linked to the ``neighbours`` other samples it correlates with
    most. An edge is kept wherever either of the two names the other, so the
    result is symmetric and some samples end up with more links than
    ``neighbours``. Ties at the last place are broken by position in the file,
    which makes the same counts give the same network every time.

    :param counts: samples down the rows, genes across the columns.
    :param neighbours: how many other samples each one names. Ten by default,
        which is the number the project's own cohort was measured at.
    :param mutual: keep an edge only where each of the two samples is among
        the other's ``neighbours``. Off by default. It gives a sparser network
        in which some samples have no links at all.
    :param correlation: a correlation matrix computed already, so that trying
        several numbers of neighbours does not compute it each time. The
        counts are then not read at all.
    :returns: a square symmetric matrix of zeros and ones, one row and column
        per sample, with no sample linked to itself.
    :raises DataError: if ``neighbours`` is below one or leaves every sample
        linked to every other, or if the network comes out with no edges. A
        network with no edges is the failure that reached a full training run
        unnoticed (finding 35); a complete one says nothing, because every
        sample then attends to the whole cohort.
    """
    similarity = sample_correlations(counts) if correlation is None else np.asarray(correlation)
    n = similarity.shape[0]
    if similarity.shape != (n, n):
        raise DataError(
            f"the correlation matrix is shaped {similarity.shape}; it has to be "
            f"square with one row per sample"
        )

    if neighbours < 1:
        raise DataError(
            f"asked for {neighbours} neighbour(s) per sample, which leaves a "
            f"network with no edges. A graph model over unconnected samples has "
            f"nothing to attend to. Ask for at least 1, or use the simple model."
        )
    if neighbours >= n - 1:
        raise DataError(
            f"asked for {neighbours} neighbour(s) per sample over {n} samples, "
            f"which links every sample to every other. A complete network says "
            f"nothing about which samples resemble which. Ask for fewer than "
            f"{n - 1}."
        )

    scores = np.array(similarity, dtype=np.float64)
    np.fill_diagonal(scores, -np.inf)
    nearest = np.argsort(-scores, axis=1)[:, :neighbours]
    directed = np.zeros((n, n), dtype=bool)
    directed[np.repeat(np.arange(n), neighbours), nearest.ravel()] = True
    network = (directed & directed.T) if mutual else (directed | directed.T)
    network = network.astype(np.int8)

    edges = int(np.count_nonzero(np.triu(network, 1)))
    if edges == 0:
        raise DataError(
            f"{neighbours} mutual neighbour(s) leaves this cohort with no edges "
            f"at all: no two samples are among each other's {neighbours} most "
            f"similar. Ask for more neighbours, or leave mutual off."
        )
    if edges == n * (n - 1) // 2:
        raise DataError(
            f"{neighbours} neighbour(s) over {n} samples leaves every pair of "
            f"samples linked, which says nothing about which samples resemble "
            f"which. Ask for fewer."
        )
    return network


@dataclass(frozen=True, slots=True)
class NetworkSummary:
    """What a sample-sample network looks like, in the numbers worth printing.

    Returned by :func:`describe_sample_network` so that a caller can print it,
    put it in a report, or check it against a number recorded earlier, without
    rebuilding the network to ask a second question of it.
    """

    #: How many samples the network covers.
    samples: int
    #: How many links there are, counting each one once rather than twice.
    edges: int
    #: The smallest, middle and largest number of links a sample has.
    degree_min: int
    degree_median: float
    degree_max: int
    #: Samples with no link at all. A graph model learns nothing for these.
    isolated_samples: int
    #: The share of links joining two samples with the same label, and the
    #: share that would be expected if the links ignored the label entirely.
    #: Both None when no labels were given. The gap between them is what says
    #: the network carries biology rather than noise.
    same_label_fraction: float | None = None
    same_label_by_chance: float | None = None
    #: What the labels are, for the sentence :meth:`describe` writes.
    label_name: str | None = None

    @property
    def edges_per_sample(self) -> float:
        """Links per sample, which is each sample's average number of links."""
        return 2 * self.edges / self.samples if self.samples else 0.0

    def describe(self) -> str:
        """Two lines a command or an interface can print as they are."""
        isolated = (
            "no isolated samples"
            if self.isolated_samples == 0
            else f"{self.isolated_samples} sample(s) with no link at all"
        )
        lines = [
            f"{self.samples} samples, {self.edges:,} edge(s), "
            f"{self.edges_per_sample:.1f} per sample "
            f"(fewest {self.degree_min}, middle {self.degree_median:.0f}, "
            f"most {self.degree_max}); {isolated}"
        ]
        if self.same_label_fraction is not None:
            lines.append(
                f"{self.same_label_fraction:.1%} of edges join two samples with "
                f"the same {self.label_name}, against {self.same_label_by_chance:.1%} "
                f"if the edges ignored it"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def describe_sample_network(
    network: Any,
    labels: Any = None,
    label_name: str = "label",
) -> NetworkSummary:
    """Measure a network: how many links, how they are spread, what they join.

    :param network: a square symmetric matrix of zeros and ones, sparse or
        dense.
    :param labels: one label per sample, such as a subtype or a batch. Given
        these, the summary also says what share of the links join two samples
        with the same label and what share would do so if the links ignored
        the label. Leave it unset and those two are None.
    :param label_name: what to call the labels in the printed sentence.
    :returns: the summary.
    :raises DataError: if the network is not square, not symmetric, links a
        sample to itself, or has a number of labels that is not the number of
        samples.
    """
    dense = _dense_network(network)
    n = dense.shape[0]
    problems = _problems_with_network(dense, n)
    if problems:
        raise DataError(
            f"{len(problems)} problem(s) with the network:\n  - " + "\n  - ".join(problems)
        )

    degree = dense.sum(axis=1)
    rows, cols = np.nonzero(np.triu(dense, 1))

    same: float | None = None
    chance: float | None = None
    if labels is not None:
        values = np.asarray(labels, dtype=object).ravel()
        if values.size != n:
            raise DataError(
                f"the network covers {n} samples but {values.size} label(s) were "
                f"given; there has to be one label per sample, in the order the "
                f"rows are in"
            )
        same = float(np.mean(values[rows] == values[cols])) if rows.size else float("nan")
        chance = _chance_same_label(values)

    return NetworkSummary(
        samples=int(n),
        edges=int(rows.size),
        degree_min=int(degree.min()),
        degree_median=float(np.median(degree)),
        degree_max=int(degree.max()),
        isolated_samples=int((degree == 0).sum()),
        same_label_fraction=same,
        same_label_by_chance=chance,
        label_name=label_name,
    )


def _chance_same_label(labels: np.ndarray) -> float:
    """The share of all pairs of samples that carry the same label."""
    n = len(labels)
    _, sizes = np.unique(labels.astype(str), return_counts=True)
    if n < 2:
        return float("nan")
    return float(sum(int(s) * (int(s) - 1) for s in sizes) / (n * (n - 1)))


# ---------------------------------------------------------------------------
# Building a correspondence table
# ---------------------------------------------------------------------------


def correspondence_table(
    sample_tables: Sequence[Any],
    patient_column: str,
) -> pd.DataFrame:
    """Say which sample of one tissue is the same patient as which sample of another.

    The full model takes several profiles from the same patient and uses them
    together, so it has to know which row of one file matches which row of
    another. The files do not share sample names, in this project's own data
    and usually elsewhere, so the correspondence is written down instead. This
    builds that table from the one thing the tissues do share: a column naming
    the patient each sample came from.

    :param sample_tables: one table per tissue, in tissue order, each with the
        tissue's sample names as its index and a column naming the patient.
    :param patient_column: the name of that column, the same in every table.
    :returns: one row per patient and one column per tissue, called ``tau_1``,
        ``tau_2`` and so on, holding that tissue's name for the patient. A
        patient with no sample in a tissue gets the string ``NA`` there, which
        is what the reader treats as missing. Patients appear in the order
        they are first seen, starting with the first tissue.
    :raises DataError: if no tables were given, if a table has no such column,
        if a sample has no patient recorded, if a sample name is repeated
        within a tissue, or if a patient appears twice in one tissue. The last
        is refused because a patient has one sample per tissue by definition,
        and a table that says otherwise makes the alignment repeat one row and
        drop another without saying so.
    """
    import pandas as pd

    tables = list(sample_tables)
    if not tables:
        raise DataError("a correspondence table needs at least one tissue's samples")

    per_tissue: list[dict[str, str]] = []
    order: list[str] = []
    seen_patients: set[str] = set()
    for index, table in enumerate(tables, start=1):
        frame = _as_sample_frame(table, f"tissue {index}")
        if patient_column not in frame.columns:
            raise DataError(
                f"tissue {index}'s sample table has no column called "
                f"{patient_column!r}, which is what says two samples are the same "
                f"patient. It has: {list(frame.columns) or '(no columns)'}"
            )
        names = [str(name) for name in frame.index]
        repeated = _repeats(names)
        if repeated:
            raise DataError(
                f"tissue {index} lists {len(repeated)} sample name(s) more than "
                f"once: {_first_few(repeated)}. A sample name has to be unique "
                f"within a tissue."
            )

        patients = frame[patient_column]
        missing = [names[i] for i, value in enumerate(patients) if _is_missing(value)]
        if missing:
            raise DataError(
                f"tissue {index} records no patient for {len(missing)} sample(s): "
                f"{_first_few(missing)}. Fill the {patient_column!r} column in, or "
                f"take those samples out."
            )

        mapping: dict[str, str] = {}
        twice: list[str] = []
        for name, patient in zip(names, patients, strict=True):
            key = str(patient)
            if key == MISSING_COUNTERPART:
                raise DataError(
                    f"tissue {index} names a patient {MISSING_COUNTERPART!r} for "
                    f"sample {name}, and that is the word the table uses for a "
                    f"patient who has no sample in a tissue. Rename the patient."
                )
            if key in mapping:
                twice.append(key)
            else:
                mapping[key] = name
                if key not in seen_patients:
                    seen_patients.add(key)
                    order.append(key)
        if twice:
            raise DataError(
                f"tissue {index} has more than one sample for {len(twice)} "
                f"patient(s): {_first_few(sorted(set(twice)))}. A patient has one "
                f"sample per tissue, and a table that says otherwise makes the "
                f"full model repeat one row and drop another without saying so. "
                f"Split them into separate tissues, or keep one sample each."
            )
        per_tissue.append(mapping)

    # Patients that only later tissues know about still get a row, with NA in
    # the tissues they are missing from. That is the case the full model pads.
    columns = {
        f"tau_{index}": [mapping.get(patient, MISSING_COUNTERPART) for patient in order]
        for index, mapping in enumerate(per_tissue, start=1)
    }
    table = pd.DataFrame(columns, index=pd.Index(order, dtype=object, name=patient_column))
    return table.astype(object)


# ---------------------------------------------------------------------------
# Writing a tissue file
# ---------------------------------------------------------------------------


def tissue_file_name(tissue: int, stem: str = "counts") -> str:
    """The file name the loader looks for, for one tissue.

    ``counts.tau_1.h5ad`` for the first tissue. The number at the end is what
    the loader matches on, so nothing may follow it.

    :raises DataError: if the tissue number is below one.
    """
    if tissue < 1:
        raise DataError(f"tissue numbers count from 1, and this one is {tissue}")
    return f"{stem}.tau_{tissue}.h5ad"


def write_tissue(
    folder: Any,
    counts: Any,
    samples: Any,
    genes: Any = None,
    *,
    tissue: int = 1,
    stem: str = "counts",
    network: Any = None,
    association: Any = None,
    overwrite: bool = False,
) -> Path:
    """Write one tissue's counts and annotations as a file ``preffect check`` accepts.

    Everything the file needs is checked before anything is written, and every
    problem found is reported in one message rather than one run at a time.

    :param folder: where the file goes. Created if it is not there.
    :param counts: samples down the rows, genes across the columns. An array,
        or a data frame whose index is the sample names and whose columns are
        the gene names.
    :param samples: what is known about each sample, as a data frame indexed by
        the sample names. A plain list of names is accepted for a study that
        has nothing else to record. When the counts are a data frame with
        sample names of their own, the two are matched by name rather than by
        position, and any disagreement between them is refused.
    :param genes: the gene names. May be left unset when ``counts`` is a data
        frame, in which case its columns are used.
    :param tissue: the tissue number, counting from 1, which goes into the file
        name.
    :param stem: what the file is called before the tissue number.
    :param network: the sample-sample network, square and symmetric with no
        sample linked to itself, in the row order of the counts.
    :param association: the correspondence table from
        :func:`correspondence_table`, covering every tissue. The rows belonging
        to this tissue are taken out of it and written under the tissue's own
        sample names, which is the shape the full model reads.
    :param overwrite: write over a file that is already there. Off by default.
    :returns: the path written.
    :raises DataError: if the counts are not whole numbers that are not
        negative, naming the entries at fault; if a sample name or a gene name
        is repeated; if the shapes disagree; if the network is not square,
        not symmetric, has a sample linked to itself or has no edges; if the
        correspondence table does not name this tissue's samples once each; or
        if the file is there already and ``overwrite`` is off.
    """
    import anndata as ad
    import pandas as pd

    folder = Path(folder)
    path = folder / tissue_file_name(tissue, stem)
    if path.exists() and not overwrite:
        raise DataError(
            f"{path} is there already, and writing over it would lose whatever it "
            f"holds. Write to another folder, or say that replacing it is what "
            f"you meant: --overwrite on the command line, overwrite=True here."
        )

    sample_frame = _as_sample_frame(samples, "the sample table")
    if isinstance(counts, pd.DataFrame) and not isinstance(counts.index, pd.RangeIndex):
        # The counts carry the sample names too, so the two are matched by
        # name rather than by position. Position would quietly give a sample
        # someone else's annotations when the two files are sorted
        # differently, which is the sort of thing that reaches a result.
        sample_frame = _match_to_counts(sample_frame, [str(n) for n in counts.index])
    sample_names = [str(name) for name in sample_frame.index]
    matrix, gene_names = _counts_and_genes(counts, genes, len(sample_names))

    problems = _problems_with_names(sample_names, "sample")
    problems += _problems_with_names(gene_names, "gene")
    if matrix.shape[1] != len(gene_names):
        # The entry-by-entry checks name each bad entry by its gene, so they
        # only mean anything once there is a name per column.
        problems.append(
            f"the counts have {matrix.shape[1]} column(s) and {len(gene_names)} "
            f"gene name(s) were given; there has to be one name per column"
        )
    else:
        problems += _problems_with_counts_matrix(matrix, sample_names, gene_names)
    if problems:
        raise DataError(
            f"{len(problems)} problem(s) with tissue {tissue}:\n  - "
            + "\n  - ".join(problems)
        )

    adata = ad.AnnData(np.ascontiguousarray(matrix, dtype=np.float32))
    # Object dtype, as the rest of the package writes names: under pandas 3 a
    # plain list becomes a string array that anndata 0.11 refuses to write,
    # and a fresh install can resolve exactly that pairing. Finding 48.
    adata.obs_names = pd.Index(sample_names, dtype=object)
    adata.var_names = pd.Index(gene_names, dtype=object)
    for column in sample_frame.columns:
        adata.obs[column] = sample_frame[column].to_numpy()

    if network is not None:
        adata.obsm["sample_sample_adj"] = _network_for_file(network, sample_names)
    if association is not None:
        adata.obsm["sample_association"] = _association_for_tissue(
            association, sample_names, tissue
        )

    folder.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path


@dataclass(frozen=True, slots=True)
class WrittenNetwork:
    """A copy of a tissue file with a network in it, and what the network looks like."""

    #: The file written. The file read is left as it was.
    path: Path
    #: Edges, spread, and the same-label share, from :func:`describe_sample_network`.
    summary: NetworkSummary
    #: True when the file read already had a network entry that held no edges
    #: between different samples, which is what every dataset version 0.1
    #: shipped carries and what this replaced.
    replaced_an_empty_network: bool

    def describe(self) -> str:
        """What happened and what the network looks like, ready to print."""
        first = f"wrote {self.path}"
        if self.replaced_an_empty_network:
            first += ", replacing a network entry that had no edges in it"
        return "\n".join([first, self.summary.describe()])


def add_network_to_file(
    path: Any,
    out: Any = None,
    *,
    neighbours: int = DEFAULT_NEIGHBOURS,
    mutual: bool = False,
    label_column: str | None = None,
    replace: bool = False,
    overwrite: bool = False,
) -> WrittenNetwork:
    """Copy a tissue file, adding a sample-sample network built from its own counts.

    For the common case: the file exists already and only the network is
    missing. The file read is never changed, which is both the project's rule
    for source data and the only safe thing to do to somebody's only copy.

    :param path: the tissue file to read.
    :param out: where the copy goes. Left unset it is written beside the
        original, with ``_with_network`` added before the tissue number so the
        loader still recognises it.
    :param neighbours: how many other samples each one names. Ten by default.
    :param mutual: keep an edge only where each sample is among the other's
        neighbours.
    :param label_column: a sample annotation, such as a subtype, to measure the
        network against. The summary then says what share of the links join
        two samples with the same value of it.
    :param replace: replace a network that is already there and has edges in
        it. Off by default, so a file that already carries a real network is
        refused rather than quietly rebuilt. A network entry with no edges
        between different samples is replaced whatever this says, because it
        is not a network.
    :param overwrite: write over an output file that is there already.
    :returns: the path written and what the network looks like.
    :raises DataError: for anything :func:`nearest_neighbour_network` refuses,
        if the file name does not end in ``tau_<n>.h5ad``, if the output is
        there already, or if the file carries a network with edges and
        ``replace`` is off.
    """
    import anndata as ad

    from preffect.data.loading import TISSUE_FILE_PATTERN

    path = Path(path)
    if TISSUE_FILE_PATTERN.search(path.name) is None:
        raise DataError(
            f"{path.name} does not end in 'tau_<n>.h5ad', so the loader would not "
            f"find it and neither does this. Rename it, for example to "
            f"'{path.stem}.tau_1.h5ad'."
        )
    if out is not None:
        destination = Path(out)
    else:
        # Beside the original, with the tissue number still at the end, since
        # that is what the loader matches on.
        match = TISSUE_FILE_PATTERN.search(path.name)
        before = path.name[: match.start()].rstrip("._")
        destination = path.with_name(
            f"{before + '_' if before else ''}with_network.tau_{match.group(1)}.h5ad"
        )
    if destination.exists() and not overwrite:
        raise DataError(
            f"{destination} is there already, and writing over it would lose "
            f"whatever it holds. Give another name, or say that replacing it is "
            f"what you meant: --overwrite on the command line, overwrite=True here."
        )

    adata = ad.read_h5ad(path)
    # Read before the new network goes in, because afterwards the entry is
    # there whatever was there before.
    had_an_entry = "sample_sample_adj" in adata.obsm
    had_edges = _existing_edges(adata)
    if had_edges and not replace:
        raise DataError(
            f"{path.name} already carries a sample-sample network with "
            f"{had_edges:,} edge(s). Use it, or ask for a new one to be built "
            f"from the counts: --replace on the command line, replace=True here."
        )

    labels = None
    if label_column is not None:
        if label_column not in adata.obs.columns:
            raise DataError(
                f"{path.name} has no sample annotation called {label_column!r} to "
                f"measure the network against. It has: "
                f"{list(adata.obs.columns) or '(nothing)'}"
            )
        labels = adata.obs[label_column].to_numpy()

    counts = _dense_counts(adata.X)
    network = nearest_neighbour_network(counts, neighbours, mutual=mutual)
    summary = describe_sample_network(network, labels, label_column or "label")

    adata.obsm["sample_sample_adj"] = _sparse_int8(network)
    destination.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(destination)
    return WrittenNetwork(
        path=destination,
        summary=summary,
        replaced_an_empty_network=had_an_entry and had_edges == 0,
    )


# ---------------------------------------------------------------------------
# Reading the tables a study already has
# ---------------------------------------------------------------------------

#: What the counts table is assumed to hold, and the other way round.
ORIENTATIONS = ("samples-by-genes", "genes-by-samples")


def read_counts_table(source: Any, orientation: str = ORIENTATIONS[0]) -> pd.DataFrame:
    """Read a counts table from a comma- or tab-separated file.

    **It assumes samples down the rows and genes across the columns**, with the
    first column holding the sample names and the first row the gene names.
    A table the other way round, which is how many pipelines write one, is read
    with ``orientation="genes-by-samples"`` and flipped here.

    :param source: a path, or anything pandas can read, such as the buffer a
        browser upload arrives in. A path ending ``.tsv``, ``.tab`` or ``.txt``
        is read as tab-separated and anything else as comma-separated;
        ``.gz`` on the end is unpacked first.
    :param orientation: ``samples-by-genes`` or ``genes-by-samples``.
    :returns: a data frame with samples down the rows and genes across the
        columns, whichever way the file was written.
    :raises DataError: if the orientation is not one of the two, or the table
        has no rows or no columns.
    """
    if orientation not in ORIENTATIONS:
        raise DataError(
            f"orientation is {orientation!r}; it has to be one of "
            f"{list(ORIENTATIONS)}. The first means samples down the rows."
        )
    frame = _read_separated(source, "the counts table", index_col=0)
    if orientation == "genes-by-samples":
        frame = frame.T
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise DataError(
            f"the counts table read as {frame.shape[0]} row(s) by "
            f"{frame.shape[1]} column(s). Read as {orientation}, that is "
            f"{frame.shape[0]} sample(s) and {frame.shape[1]} gene(s)."
        )
    frame.index = frame.index.astype(object)
    frame.columns = frame.columns.astype(object)
    return frame


def read_sample_table(source: Any, sample_column: str | None = None) -> pd.DataFrame:
    """Read a table of sample annotations from a comma- or tab-separated file.

    **It assumes the first column holds the sample names**, and those names
    have to be the ones used in the counts table. Name another column with
    ``sample_column`` when the file puts the names somewhere else.

    :param source: a path, or anything pandas can read.
    :param sample_column: which column holds the sample names. The first
        column by default.
    :returns: a data frame indexed by the sample names.
    :raises DataError: if the named column is not in the file.
    """
    frame = _read_separated(source, "the sample table")
    if sample_column is None:
        if frame.shape[1] == 0:
            raise DataError("the sample table has no columns at all")
        sample_column = str(frame.columns[0])
    if sample_column not in frame.columns:
        raise DataError(
            f"the sample table has no column called {sample_column!r} to take the "
            f"sample names from. It has: {list(frame.columns)}"
        )
    frame = frame.set_index(sample_column)
    frame.index = frame.index.astype(object)
    return frame


def _read_separated(source: Any, what: str, **options):
    """Read a separated-values table, refusing in a sentence rather than a traceback.

    pandas raises an ``OSError`` for a path that is not there, and its own
    parser errors for a file that is not a table at all. Both of those reach
    somebody who did not write this code, as twenty lines of stack ending in
    the name of a pandas internal, and the command line prints them whole
    because it only catches this package's own errors. Finding 66.

    :param what: which table is being read, for the message.
    :raises DataError: if the file cannot be opened or cannot be read as a
        table of separated values.
    """
    import pandas as pd

    name = getattr(source, "name", source)
    try:
        return pd.read_csv(source, sep=_separator(source), **options)
    except OSError as error:
        raise DataError(
            f"could not open {what} at {name}: {error}. Check the path, and "
            f"that you are running where you think you are."
        ) from error
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise DataError(
            f"could not read {what} at {name} as a table: {error}. It has to be "
            f"text with one value per column, separated by commas, or by tabs "
            f"when the name ends .tsv, .tab or .txt."
        ) from error


def _separator(source: Any) -> str:
    """Tab for a file named like a tab-separated one, comma otherwise."""
    name = str(getattr(source, "name", source)).lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return "\t" if name.endswith((".tsv", ".tab", ".txt")) else ","


# ---------------------------------------------------------------------------
# The checks the writer runs, each reporting what is wrong rather than fixing it
# ---------------------------------------------------------------------------


def _as_array(values: Any, what: str) -> np.ndarray:
    """A dense float array from an array, a data frame or a sparse matrix."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    elif hasattr(values, "todense"):
        values = values.todense()
    array = np.asarray(values)
    if array.dtype.kind not in "fiub":
        raise DataError(
            f"{what} hold values of type {array.dtype}, which are not numbers. "
            f"A column of text where a count was expected is the usual cause."
        )
    return array.astype(np.float64, copy=False)


def _dense_counts(matrix: Any) -> np.ndarray:
    """One tissue's counts as a dense array, whether they were stored sparse or not."""
    return np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix)


def _as_sample_frame(samples: Any, what: str) -> pd.DataFrame:
    """A data frame indexed by sample name, from a frame or a plain list of names."""
    import pandas as pd

    if isinstance(samples, pd.DataFrame):
        frame = samples
    elif isinstance(samples, pd.Series):
        frame = samples.to_frame()
    else:
        names = [str(name) for name in samples]
        frame = pd.DataFrame(index=pd.Index(names, dtype=object))
    if isinstance(frame.index, pd.RangeIndex):
        raise DataError(
            f"{what} is numbered 0, 1, 2 and so on rather than indexed by the "
            f"sample names, so there is nothing to match the counts against. "
            f"Set the index to the sample names, for example with "
            f"table.set_index('sample')."
        )
    return frame


def _match_to_counts(frame: pd.DataFrame, counts_names: list[str]) -> pd.DataFrame:
    """Put the sample table in the counts table's row order, matching by name.

    Returns the frame untouched when the two already agree, which is the
    ordinary case and the one worth keeping cheap.
    """
    own = [str(name) for name in frame.index]
    if own == counts_names:
        return frame

    for names, what in ((own, "sample table"), (counts_names, "counts")):
        repeated = _repeats(names)
        if repeated:
            raise DataError(
                f"the {what} lists {len(repeated)} sample name(s) more than once "
                f"({_first_few(repeated)}), so the two cannot be matched up by "
                f"name. Make the names unique."
            )

    in_table, in_counts = set(own), set(counts_names)
    only_counts = [name for name in counts_names if name not in in_table]
    only_table = [name for name in own if name not in in_counts]
    if only_counts or only_table:
        parts = []
        if only_counts:
            parts.append(
                f"{len(only_counts)} sample(s) in the counts are not in the sample "
                f"table: {_first_few(only_counts)}"
            )
        if only_table:
            parts.append(
                f"{len(only_table)} sample(s) in the sample table are not in the "
                f"counts: {_first_few(only_table)}"
            )
        raise DataError(
            "the counts and the sample table describe different samples. "
            + "; ".join(parts)
            + ". They have to name the same samples."
        )
    return frame.loc[counts_names]


def _counts_and_genes(counts: Any, genes: Any, n_samples: int) -> tuple[np.ndarray, list[str]]:
    """The counts as an array and the gene names, from whichever carries them."""
    import pandas as pd

    names: list[str] | None = None
    if isinstance(counts, pd.DataFrame):
        names = [str(c) for c in counts.columns]
    if genes is not None:
        names = [str(g) for g in genes]
    if names is None:
        raise DataError(
            "no gene names were given and the counts are a plain array, so there "
            "is nothing to call the columns. A gene name is the key selection and "
            "every later step match on, so they cannot be made up here. Pass "
            "genes=, or give the counts as a data frame whose columns are the "
            "gene names."
        )

    matrix = _as_array(counts, "the counts")
    if matrix.ndim != 2:
        raise DataError(
            f"the counts have {matrix.ndim} dimension(s); they must be a table "
            f"with samples down the rows and genes across the columns"
        )
    if matrix.shape[0] != n_samples:
        raise DataError(
            f"the counts have {matrix.shape[0]} row(s) and the sample table has "
            f"{n_samples} row(s). If the counts are genes down the rows and "
            f"samples across the columns, which is how many pipelines write "
            f"them, transpose them first."
        )
    return matrix, names


def _problems_with_counts_matrix(
    matrix: np.ndarray, sample_names: list[str], gene_names: list[str]
) -> list[str]:
    """Every entry that is not a count, said in a way that points at the entry."""
    problems: list[str] = []
    if matrix.size == 0:
        return ["the counts are empty"]

    missing = np.argwhere(np.isnan(matrix))
    if missing.size:
        problems.append(
            f"{len(missing)} count(s) are missing, at "
            f"{_entries(missing, sample_names, gene_names, matrix)}. A count model "
            f"cannot use them; set them to 0, or drop the gene or sample."
        )
    finite = np.isfinite(matrix)
    negative = np.argwhere(finite & (matrix < 0))
    if negative.size:
        problems.append(
            f"{len(negative)} count(s) are negative, at "
            f"{_entries(negative, sample_names, gene_names, matrix)}. This usually "
            f"means the table holds transformed values rather than raw counts."
        )
    fractional = np.argwhere(finite & (np.mod(matrix, 1) != 0))
    if fractional.size:
        problems.append(
            f"{len(fractional)} count(s) are not whole numbers, at "
            f"{_entries(fractional, sample_names, gene_names, matrix)}. PREFFECT "
            f"models counts, so normalised or log-transformed values will not fit."
        )
    return problems


def _entries(
    positions: np.ndarray,
    sample_names: list[str],
    gene_names: list[str],
    matrix: np.ndarray,
    how_many: int = 3,
) -> str:
    """Name the first few offending entries by sample, gene and value."""
    shown = [
        f"{sample_names[int(i)]}/{gene_names[int(j)]}={matrix[int(i), int(j)]:g}"
        for i, j in positions[:how_many]
    ]
    more = f" and {len(positions) - how_many} more" if len(positions) > how_many else ""
    return ", ".join(shown) + more


def _problems_with_names(names: list[str], kind: str) -> list[str]:
    """Repeated names, which silently drop a row or pull several columns for one."""
    repeated = _repeats(names)
    if not repeated:
        return []
    return [
        f"{len(repeated)} {kind} name(s) appear more than once "
        f"({_first_few(repeated)}). A {kind} name is the key everything matches "
        f"on, so a repeat quietly loses one of them."
    ]


def _repeats(names: list[str]) -> list[str]:
    """The names that appear more than once, in the order they first appear."""
    seen: dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    return [name for name, count in seen.items() if count > 1]


def _first_few(values: Sequence[str], how_many: int = 5) -> str:
    """The first few of a list, with a count of what is left."""
    shown = ", ".join(str(v) for v in values[:how_many])
    return shown + (f" and {len(values) - how_many} more" if len(values) > how_many else "")


def _is_missing(value: Any) -> bool:
    """A patient identifier that is absent, empty, or the word for absent."""
    import pandas as pd

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _dense_network(network: Any) -> np.ndarray:
    """A network as a dense integer array, whether it was stored sparse or not."""
    dense = np.asarray(network.todense() if hasattr(network, "todense") else network)
    if dense.ndim != 2:
        raise DataError(
            f"the network has {dense.ndim} dimension(s); it has to be a square "
            f"matrix with one row and column per sample"
        )
    return np.array(dense)


def _problems_with_network(dense: np.ndarray, n: int) -> list[str]:
    """Everything that would make a network unusable, in the order worth saying."""
    problems: list[str] = []
    if dense.shape != (n, n):
        return [
            f"the network is shaped {dense.shape}; it has to be square with one "
            f"row and column per sample"
        ]
    self_links = int(np.count_nonzero(np.diag(dense)))
    if self_links:
        problems.append(
            f"the network links {self_links} sample(s) to themselves. Every sample "
            f"attends to itself anyway, and counting self-links flatters any "
            f"measure of how well the network is reproduced. Set the diagonal to "
            f"zero. A network that is only a diagonal is what every dataset "
            f"version 0.1 shipped carries, and it holds no edges at all "
            f"(findings 35 and 54)."
        )
    if not np.array_equal(dense, dense.T):
        problems.append(
            "the network is not symmetric, so it says sample A resembles sample B "
            "without saying the reverse. Make it symmetric, for example by "
            "keeping an edge wherever either direction has one."
        )
    if not np.isin(dense, (0, 1)).all():
        problems.append(
            "the network holds values other than 0 and 1. PREFFECT reads it as a "
            "list of links, so a weight on a link is ignored; threshold it first."
        )
    return problems


def _network_for_file(network: Any, sample_names: list[str]) -> Any:
    """A network checked against the samples it covers, stored small."""
    dense = _dense_network(network)
    n = len(sample_names)
    problems = _problems_with_network(dense, n)
    if problems:
        raise DataError(
            f"{len(problems)} problem(s) with the sample-sample network:\n  - "
            + "\n  - ".join(problems)
        )
    if int(np.count_nonzero(np.triu(dense, 1))) == 0:
        raise DataError(
            "the sample-sample network has no edges between different samples, so "
            "a graph model over it has nothing to attend to. Leave it out and use "
            "the simple model, or build one with nearest_neighbour_network."
        )
    return _sparse_int8(dense)


def _sparse_int8(dense: np.ndarray) -> Any:
    """A network as a sparse matrix, which is how the project's files store it."""
    from scipy.sparse import csr_matrix

    return csr_matrix(dense.astype(np.int8))


def _existing_edges(adata: ad.AnnData) -> int:
    """How many edges between different samples a file's network entry holds."""
    if "sample_sample_adj" not in adata.obsm:
        return 0
    dense = _dense_network(adata.obsm["sample_sample_adj"])
    np.fill_diagonal(dense, 0)
    return int(np.count_nonzero(dense)) // 2


def _association_for_tissue(association: Any, sample_names: list[str], tissue: int) -> pd.DataFrame:
    """One tissue's rows of the correspondence table, under that tissue's own names.

    A tissue's copy of the table holds the rows for the patients that tissue
    actually has, indexed by its own sample names and in the order its rows
    are in. That is the shape the alignment reads, and the shape anndata
    requires of a table stored beside the samples.
    """
    import pandas as pd

    if not isinstance(association, pd.DataFrame):
        raise DataError(
            f"the correspondence table is a {type(association).__name__}; it has "
            f"to be a data frame with one column per tissue, as "
            f"correspondence_table returns."
        )
    column = f"tau_{tissue}"
    if column not in association.columns:
        raise DataError(
            f"the correspondence table has no column called {column!r} for tissue "
            f"{tissue}. It has: {list(association.columns)}"
        )

    table = association.astype(object)
    named = [str(v) for v in table[column] if str(v) != MISSING_COUNTERPART]
    repeated = _repeats(named)
    if repeated:
        raise DataError(
            f"the correspondence table names {len(repeated)} sample(s) of tissue "
            f"{tissue} for more than one patient: {_first_few(repeated)}. A sample "
            f"belongs to one patient."
        )
    in_table, in_file = set(named), set(sample_names)
    missing_from_table = [name for name in sample_names if name not in in_table]
    if missing_from_table:
        raise DataError(
            f"the correspondence table never names {len(missing_from_table)} of "
            f"tissue {tissue}'s samples: {_first_few(missing_from_table)}. "
            f"Aligning would drop them without saying so. Add them to the table, "
            f"or take them out of the counts."
        )
    absent = [name for name in named if name not in in_file]
    if absent:
        raise DataError(
            f"the correspondence table names {len(absent)} sample(s) that tissue "
            f"{tissue} does not have: {_first_few(absent)}."
        )

    rows = table[[str(v) != MISSING_COUNTERPART for v in table[column]]].copy()
    rows.index = pd.Index([str(v) for v in rows[column]], dtype=object)
    return rows.loc[sample_names]
