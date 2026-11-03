"""A small simulated cohort to try PREFFECT on, written by ``preffect example-data``.

Three splits with no sample in common, for each of the three models: the
simple model, the single model (the same counts plus a sample-sample network
with real edges in it) and the full model (two tissues per patient, under
different names, with the correspondence written down). About two seconds,
200 genes, three groups of samples that each raise a different block of
genes, so there is something real to find and you can tell whether the model
found it. The counts are negative binomial, which is what PREFFECT models.

**The genes are drawn once per cohort and shared by its three splits.** Each
gene has one base level and one dispersion, and every split of that cohort is
drawn from them, so the gene that is highly expressed in the training data is
the same gene that is highly expressed in the test data. Until finding 65 the
draw happened inside :func:`simulate`, which is called once per split, so the
three splits were three unrelated gene populations sharing a set of names: a
model fitted on the training split could not track the test split, and the
validation loss sat far above the training loss for that reason rather than
for any reason about PREFFECT.

**Version 0.1's vignettes ship a ``test/`` folder byte-identical to
``validation/``**, in all three of simple, single and full, and a published
figure shows a user pointing inference at that folder while the caption calls
it a held-out test set. The data here is generated rather than committed, for
two reasons: counts do not belong in a git repository, and a generator can
promise what a committed file cannot. The three splits are drawn from disjoint
samples by construction, and :func:`write_example_data` checks that and raises
rather than write them if they are not. Issue #11.

**Every file carries the true expected count of every entry**, in
``layers['true_expected_counts']``: the negative binomial mean each count was
drawn from, before the draw added its noise. Nothing in PREFFECT reads it, and
no real cohort can have it. It is there so that an example which fills in
hidden entries can be scored against what the entry should have been rather
than against one noisy draw of it, and so that a reader can tell at a glance
that the file is simulated. ``preffect check`` says so when it finds it.

**Batch B's genes are really raised.** The files have always carried a batch
column, and until finding 65 it labelled samples that were drawn identically,
so no correction could change anything. Now a fixed share of the genes, drawn
once with the rest of the population and so the same genes in every split, are
multiplied by :data:`BATCH_LIFT` in every batch-B sample. It is deliberately
smaller than the groups' four-fold lift, so the groups still dominate and the
batch is the second thing you see. What it demonstrates is the correction
variables, which take a known nuisance label out of the encoder's input, and
the interface's batch figure. It does not exercise the objective's
batch-centroid term, which training never reaches because no batch labels are
passed to it. Which genes moved, and by how much, is written into each file
under ``uns['batch_shifted_genes']`` and ``uns['batch_lift']``, so a reader
checking whether a correction took the batch out knows where to look.

The full model's two files per split deliberately **do not share sample
names**, as the project's own data does not, so the correspondence goes in a
table under ``obsm['sample_association']`` with one column per tissue, which
is where :func:`preffect.data.tissues.align_tissues` looks. ``unmatched`` takes
the second tissue away from a share of patients, which is the shape of a real
cohort where some patients have no matched normal, and is what the full
model's padding exists for. It is zero by default and draws no random numbers
when it is zero.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preffect.errors import ConfigError, DataError

__all__ = [
    "BATCH_LIFT",
    "BATCH_SHARE",
    "N_BATCH_GENES",
    "N_GENES",
    "N_SAMPLES",
    "N_SUBTYPES",
    "SHIFTED_BATCH",
    "TRUTH_LAYER",
    "ExampleData",
    "GenePopulation",
    "draw_population",
    "ring_network",
    "simulate",
    "write_example_data",
]

#: Samples per split. Small on purpose: an example that takes an hour is an
#: example nobody runs.
N_SAMPLES = {"train": 120, "validation": 40, "test": 40}
#: Genes in every file.
N_GENES = 200
#: Groups of samples, each raising its own block of genes.
N_SUBTYPES = 3
#: The three model kinds the data is written for, in the order it is written.
KINDS = ("simple", "single", "full")
#: The batch whose genes are raised. The other batch is left alone, so the
#: shift is a difference between the two rather than a shift of everything.
SHIFTED_BATCH = "B"
#: How much batch B raises its share of the genes. Modest against the groups'
#: four-fold lift, so a reader sees the groups first and the batch second.
BATCH_LIFT = 1.3
#: What share of a cohort's genes batch B raises.
BATCH_SHARE = 0.2
#: How many genes that is in the example data's :data:`N_GENES`.
N_BATCH_GENES = round(BATCH_SHARE * N_GENES)
#: Where every file carries the true expected count of every entry.
TRUTH_LAYER = "true_expected_counts"


@dataclass(frozen=True)
class GenePopulation:
    """The genes a cohort is drawn from, drawn once and shared by its splits.

    One of these describes a set of genes: what each is expressed at, how
    noisy each is, and which of them batch B raises. Every split of a cohort
    is drawn from the same one, which is what makes a model fitted on the
    training split able to track the test split. Finding 65.
    """

    #: Each gene's mean count before groups, sequencing depth and batch.
    base: np.ndarray
    #: Each gene's negative binomial dispersion. Smaller is noisier.
    dispersion: np.ndarray
    #: Which genes batch B raises, as column numbers.
    batch_genes: np.ndarray
    #: How much batch B raises them.
    batch_lift: float

    @property
    def n_genes(self) -> int:
        """How many genes this population describes."""
        return int(self.base.shape[0])


def draw_population(
    n_genes: int,
    rng: np.random.Generator,
    *,
    batch_lift: float = BATCH_LIFT,
    batch_share: float = BATCH_SHARE,
) -> GenePopulation:
    """Draw a set of genes for one cohort: levels, dispersions and the batch's genes.

    :param n_genes: how many genes to describe.
    :param rng: the generator; every draw comes from it, so a seed fixes the
        population and so the files.
    :param batch_lift: how much batch B raises its genes.
    :param batch_share: what share of the genes batch B raises, chosen at
        random from all of them, so some fall inside a group's block and some
        outside. A share rather than a count, so it means the same thing at
        any number of genes.
    :raises ConfigError: if the share is not above 0 and at most 1, if it
        rounds down to no genes at all, or if the lift is not positive.
    :returns: the population, to hand to :func:`simulate` once per split.
    """
    if not 0 < batch_share <= 1:
        raise ConfigError(
            f"the share of genes the batch raises has to be above 0 and at most "
            f"1, got {batch_share}"
        )
    n_batch_genes = round(batch_share * n_genes)
    if n_batch_genes == 0:
        raise ConfigError(
            f"a share of {batch_share} over {n_genes} genes rounds down to no "
            f"genes at all, so the batch would change nothing. Use a share of at "
            f"least {0.5 / n_genes:.3g}."
        )
    if batch_lift <= 0:
        raise ConfigError(
            f"the batch lift multiplies counts, so it has to be above 0, got {batch_lift}"
        )
    base = rng.lognormal(mean=3.0, sigma=1.0, size=n_genes)
    dispersion = rng.uniform(2.0, 8.0, size=n_genes)
    batch_genes = np.sort(rng.choice(n_genes, size=n_batch_genes, replace=False))
    return GenePopulation(
        base=base, dispersion=dispersion, batch_genes=batch_genes, batch_lift=float(batch_lift)
    )


def batch_labels(n_samples: int) -> list[str]:
    """Which batch each sample is in, alternating, so the two are the same size.

    The same rule the files have always used. It is a function so that the
    labels written into ``obs['batch']`` and the labels the counts were drawn
    under cannot drift apart.

    :param n_samples: rows to label.
    """
    return ["A" if i % 2 else SHIFTED_BATCH for i in range(n_samples)]


def _record_the_batch(adata, population: GenePopulation, gene_names: list[str]) -> None:
    """Write down which genes the batch raised, and by how much, in the file itself.

    By name rather than by column number, so it survives choosing a subset of
    the genes. A reader checking whether a correction took the batch out needs
    to know what to look at, and a test of the effect needs it without
    re-deriving the draw.
    """
    adata.uns["batch_lift"] = population.batch_lift
    adata.uns["batch_shifted_genes"] = [gene_names[j] for j in population.batch_genes]


@dataclass(frozen=True)
class _Draw:
    """One split's counts and everything that went into them."""

    counts: np.ndarray
    groups: np.ndarray
    names: list[str]
    #: The negative binomial mean every count was drawn from, before the draw.
    expected: np.ndarray


def _draw_split(
    n_samples: int,
    n_genes: int,
    rng: np.random.Generator,
    offset: int,
    groups: np.ndarray | None = None,
    lift: float = 4.0,
    population: GenePopulation | None = None,
    batches: list[str] | np.ndarray | None = None,
) -> _Draw:
    """Everything :func:`simulate` computes, including the means it drew from.

    Kept separate so that :func:`simulate` can return what it has always
    returned while the generator still gets the true expected counts out, and
    so that the draws stay in the order they were in: a call with no population
    and no batches consumes exactly the random numbers it consumed before
    finding 65, which is what the experiments' recorded results depend on.
    """
    if population is not None and population.n_genes != n_genes:
        raise DataError(
            f"the population describes {population.n_genes} genes and {n_genes} "
            f"were asked for; every split of a cohort has to be drawn from the "
            f"same genes as the rest of it"
        )
    if batches is not None:
        if population is None:
            raise ConfigError(
                "a batch effect needs a population, which is what says which genes "
                "the batch raises; pass the one the rest of this cohort was drawn from"
            )
        if len(batches) != n_samples:
            raise DataError(
                f"got {len(batches)} batch labels for {n_samples} samples; there has "
                f"to be one per sample"
            )

    if groups is None:
        groups = rng.integers(0, N_SUBTYPES, size=n_samples)
    if population is None:
        base = rng.lognormal(mean=3.0, sigma=1.0, size=n_genes)
    else:
        base = population.base

    means = np.tile(base, (n_samples, 1))
    block = n_genes // (N_SUBTYPES + 1)
    for group in range(N_SUBTYPES):
        rows = groups == group
        means[np.ix_(rows, np.arange(group * block, (group + 1) * block))] *= lift

    # The batch effect, drawn with the population so the same genes move in
    # every split. No random numbers of its own, so a call without it is the
    # call that was made before.
    if batches is not None:
        in_batch = np.asarray(batches) == SHIFTED_BATCH
        if in_batch.any():
            means[np.ix_(in_batch, population.batch_genes)] *= population.batch_lift

    depth = rng.lognormal(mean=0.0, sigma=0.25, size=(n_samples, 1))
    means = means * depth

    dispersion = (
        rng.uniform(2.0, 8.0, size=n_genes) if population is None else population.dispersion
    )
    probability = dispersion / (dispersion + means)
    counts = rng.negative_binomial(dispersion, probability).astype(np.float32)

    names = [f"SAMPLE_{offset + i:04d}" for i in range(n_samples)]
    return _Draw(
        counts=counts, groups=groups, names=names, expected=means.astype(np.float32)
    )


def simulate(
    n_samples: int,
    n_genes: int,
    rng: np.random.Generator,
    offset: int,
    groups: np.ndarray | None = None,
    lift: float = 4.0,
    population: GenePopulation | None = None,
    batches: list[str] | np.ndarray | None = None,
):
    """Counts with a subtype signal in them, so an example has something to find.

    Each sample belongs to one of three groups, and each group raises a
    different block of genes. Negative binomial, which is what PREFFECT models,
    so the example is not quietly fitting Gaussian data.

    **Called without a population this draws exactly what it always drew.**
    ``exp/08``, ``exp/11``, ``exp/17`` and ``exp/19`` each call it once for
    their synthetic cohorts and their recorded results depend on the draws, so
    the genes are still drawn inside, in the same order, when none are given.
    ``tests/test_example_data.py`` holds a digest of the result and fails if
    any of that moves.

    :param n_samples: rows to draw.
    :param n_genes: columns to draw.
    :param rng: the generator; every draw comes from it, so a seed fixes the file.
    :param offset: where sample numbering starts, so no name repeats across splits.
    :param groups: the group each sample belongs to. Drawn here when left
        unset; passed in for the full model's second tissue, which is the same
        patients and so must carry the same groups.
    :param lift: how much a group raises its own block of genes. Lower for a
        second tissue, which stands for the quieter normal tissue beside a
        tumour and should not simply repeat the first.
    :param population: the genes to draw from, from :func:`draw_population`.
        Left unset the genes are drawn here, which gives this split its own
        unrelated set of them: fine for a cohort that is one split, wrong for
        a cohort that is several, which is finding 65.
    :param batches: which batch each sample is in. Left unset there is no
        batch effect in the counts. Needs a population, which is what says
        which genes the batch raises.
    :raises DataError: if the population describes a different number of genes,
        or there is not one batch label per sample.
    :raises ConfigError: if batches are given without a population.
    :returns: the counts, the group of each sample, and the sample names.
    """
    drawn = _draw_split(
        n_samples, n_genes, rng, offset, groups=groups, lift=lift,
        population=population, batches=batches,
    )
    return drawn.counts, drawn.groups, drawn.names


def ring_network(n: int) -> np.ndarray:
    """A sample-sample network with real edges, for the single-model example.

    Not an identity matrix. Every real dataset version 0.1 shipped carries one
    of those, which is a network with nothing in it, and a graph model on it
    has nothing to attend to. Findings 35 and 54.

    :param n: samples; each is linked to its neighbours one and five places on.
    :returns: a symmetric ``n`` by ``n`` matrix of zeros and ones.
    """
    network = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        network[i, (i + 1) % n] = network[(i + 1) % n, i] = 1
        network[i, (i + 5) % n] = network[(i + 5) % n, i] = 1
    return network


def _advance(offset: int, n: int) -> int:
    """Where sample numbering continues after a split of ``n`` samples.

    Kept as a function so that the test of the distinct-splits guarantee can
    break it on purpose and confirm the generator refuses, rather than editing
    the generator's source text to do the same.
    """
    return offset + n


@dataclass(frozen=True)
class ExampleData:
    """What :func:`write_example_data` wrote, and the summary it prints."""

    #: The folder everything was written under.
    out: Path
    #: By model kind, then split: the first tissue's file.
    files: dict[str, dict[str, Path]]
    #: By split: how many patients had their second tissue taken away.
    unmatched: dict[str, int]

    def digest(self, kind: str, split: str) -> str:
        """The first eight hex digits of the md5 of one written file.

        :param kind: ``simple``, ``single`` or ``full``.
        :param split: ``train``, ``validation`` or ``test``.
        """
        return hashlib.md5(self.files[kind][split].read_bytes()).hexdigest()[:8]

    def describe(self) -> str:
        """The lines ``preffect example-data`` prints after writing."""
        lines = [f"wrote {self.out}"]
        for kind in KINDS:
            lines.append(f"  {kind}:")
            for split, n in N_SAMPLES.items():
                tissues = "  2 tissues" if kind == "full" else ""
                lines.append(
                    f"    {split:<11} {n:>3} samples  md5 {self.digest(kind, split)}{tissues}"
                )
        lines.append("")
        lines.append(f"{N_GENES} genes, {N_SUBTYPES} groups, negative binomial counts.")
        lines.append("No sample appears in more than one split, and the three files differ.")
        lines.append(
            "The genes are drawn once per cohort, so every split is the same genes."
        )
        lines.append(
            f"Batch {SHIFTED_BATCH} has {N_BATCH_GENES} of the {N_GENES} genes raised "
            f"{BATCH_LIFT} times, for the correction variables to take out."
        )
        lines.append(
            f"Every file carries the count each entry was drawn from in "
            f"layers['{TRUTH_LAYER}']."
        )
        if any(self.unmatched.values()):
            missing = ", ".join(f"{s} {c}" for s, c in self.unmatched.items())
            lines.append(
                f"The full model's second tissue is missing for some patients "
                f"({missing}); the full model pads a row of per-gene means in for "
                f"each of them."
            )
        else:
            lines.append(
                "The full model's two tissues are the same patients under different names."
            )
        return "\n".join(lines)


def write_example_data(out: Path, *, seed: int = 0, unmatched: float = 0.0) -> ExampleData:
    """Write the example cohort under ``out`` and check the splits are distinct.

    Writes ``<out>/{simple,single}/{train,validation,test}/counts.tau_1.h5ad``
    and, for the full model, two tissues per split,
    ``<out>/full/{train,validation,test}/counts.tau_{1,2}.h5ad``. The same seed
    writes the same files.

    The genes are drawn once per model kind, and the full model's second tissue
    gets its own draw, so within a kind the three splits are the same genes and
    a model fitted on one can be applied to another. Finding 65.

    :param out: the folder to write under; created if missing.
    :param seed: seed for every draw.
    :param unmatched: the share of patients whose second tissue is left out of
        the full model's data, at least 0 and below 1. The first tissue is never
        thinned, because it is the one the model predicts.
    :raises ConfigError: if ``unmatched`` is outside its range.
    :raises DataError: if any two splits share a sample or are the same file,
        which is the promise this exists to keep.
    :returns: what was written.
    """
    import anndata as ad
    import pandas as pd

    if not 0.0 <= unmatched < 1.0:
        raise ConfigError(f"unmatched must be at least 0 and below 1, got {unmatched}")

    out = Path(out)
    rng = np.random.default_rng(seed)
    offset = 0
    written: dict[str, dict[str, Path]] = {}
    sample_names: dict[str, list[str]] = {}

    for kind in ("simple", "single"):
        # One draw of the genes for this kind's three splits. Drawing it per
        # split, which is what happened until finding 65, made train,
        # validation and test three unrelated cohorts sharing a set of names.
        population = draw_population(N_GENES, rng)
        written[kind] = {}
        for split, n in N_SAMPLES.items():
            batches = batch_labels(n)
            drawn = _draw_split(
                n, N_GENES, rng, offset, population=population, batches=batches
            )
            offset = _advance(offset, n)  # so no sample name is ever reused across splits

            adata = ad.AnnData(drawn.counts)
            # Object dtype, as the package writes names: under pandas 3 a list
            # becomes a string array that anndata 0.11 refuses to write, and a
            # fresh install can resolve exactly that pairing. Finding 48.
            adata.obs_names = pd.Index(drawn.names, dtype=object)
            adata.var_names = pd.Index([f"GENE_{j:04d}" for j in range(N_GENES)], dtype=object)
            adata.obs["subtype"] = pd.Categorical([f"group_{g}" for g in drawn.groups])
            adata.obs["batch"] = pd.Categorical(batches)
            adata.layers[TRUTH_LAYER] = drawn.expected
            _record_the_batch(adata, population, list(adata.var_names))
            if kind == "single":
                adata.obsm["sample_sample_adj"] = ring_network(n)

            folder = out / kind / split
            folder.mkdir(parents=True, exist_ok=True)
            # The loader looks for '*tau_<n>.h5ad', where n is the tissue
            # number. An example with one tissue is tau_1.
            path = folder / "counts.tau_1.h5ad"
            adata.write_h5ad(path)
            written[kind][split] = path
            sample_names[f"{kind}/{split}"] = drawn.names

    # The full model: two tissues per split, the same patients in each, and a
    # table saying which row of one is which row of the other. Written after
    # the others so the sample names carry on rather than repeating theirs.
    # One draw of the genes per tissue: the two tissues are different tissues,
    # so they are different genes, but every split of a tissue is the same
    # genes as the other splits of that tissue.
    written["full"] = {}
    tissue_populations = [draw_population(N_GENES, rng), draw_population(N_GENES, rng)]
    full_tissue_files: dict[str, list[Path]] = {}
    unmatched_written: dict[str, int] = {}
    for split, n in N_SAMPLES.items():
        # One batch per patient, used for both of that patient's tissues: a
        # batch is a property of when the patient was processed, so it cannot
        # differ between their two samples.
        batches = np.array(batch_labels(n))
        first = _draw_split(
            n, N_GENES, rng, offset, population=tissue_populations[0], batches=batches
        )
        offset = _advance(offset, n)
        # The same patients, so the same groups; a weaker block lift, so the
        # second tissue informs the first rather than repeating it.
        second = _draw_split(
            n, N_GENES, rng, offset, groups=first.groups, lift=1.8,
            population=tissue_populations[1], batches=batches,
        )
        offset = _advance(offset, n)

        # Which patients keep their second tissue. Every one of them unless
        # asked otherwise, and the draw is skipped entirely at zero so that the
        # generator's stream, and so every file, is untouched.
        kept = [np.ones(n, dtype=bool), np.ones(n, dtype=bool)]
        n_drop = int(round(unmatched * n))
        if n_drop:
            kept[1][rng.choice(n, size=n_drop, replace=False)] = False
        unmatched_written[split] = int((~kept[1]).sum())

        # One column per tissue, each listing that tissue's name for the
        # patient on this row. The two tissues share no sample name, as the
        # project's own data does not, so this table is the only link. A
        # patient whose second tissue was taken away has an empty cell there,
        # which is how a real cohort records a missing matched normal.
        association = pd.DataFrame({"tau_1": first.names, "tau_2": second.names})
        # The empty cell is written as the string "NA", which version 0.1 used
        # and which the reader turns into a missing value, rather than as NaN:
        # under pandas 2 a column of names with a float NaN in it is an object
        # column that anndata refuses to write ("non-string objects"), while
        # pandas 3 makes it a string column and writes it. Continuous integration
        # on Python 3.11 resolved pandas 2 and failed on exactly this line.
        association.loc[~kept[1], "tau_2"] = "NA"

        folder = out / "full" / split
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        written_names = []
        for tissue, (drawn, rows, population) in enumerate(
            (
                (first, kept[0], tissue_populations[0]),
                (second, kept[1], tissue_populations[1]),
            ),
            start=1,
        ):
            own_names = [name for name, keep in zip(drawn.names, rows, strict=True) if keep]
            own_groups = drawn.groups[rows]
            m = len(own_names)
            adata = ad.AnnData(drawn.counts if rows.all() else drawn.counts[rows])
            adata.obs_names = pd.Index(own_names, dtype=object)
            adata.var_names = pd.Index([f"GENE_{j:04d}" for j in range(N_GENES)], dtype=object)
            adata.obs["subtype"] = pd.Categorical([f"group_{g}" for g in own_groups])
            adata.obs["batch"] = pd.Categorical(list(batches[rows]))
            adata.layers[TRUTH_LAYER] = (
                drawn.expected if rows.all() else drawn.expected[rows]
            )
            _record_the_batch(adata, population, list(adata.var_names))
            adata.obsm["sample_sample_adj"] = ring_network(m)
            # An obsm frame is aligned by index, so each tissue carries the
            # table under its own names, holding the rows for the patients it
            # actually has. On a complete cohort that is the whole table in
            # both files.
            own = association[rows].copy()
            own.index = pd.Index(own_names, dtype=object)
            adata.obsm["sample_association"] = own
            path = folder / f"counts.tau_{tissue}.h5ad"
            adata.write_h5ad(path)
            paths.append(path)
            written_names.extend(own_names)

        full_tissue_files[split] = paths
        written["full"][split] = paths[0]
        sample_names[f"full/{split}"] = written_names

    # The promise this exists to keep. Checked before reporting success,
    # because version 0.1's vignettes failed exactly here and nothing noticed.
    problems = []
    for kind in KINDS:
        splits = list(N_SAMPLES)
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                shared = set(sample_names[f"{kind}/{a}"]) & set(sample_names[f"{kind}/{b}"])
                if shared:
                    problems.append(f"{kind}: {a} and {b} share {len(shared)} sample(s)")
                digest_a = hashlib.md5(written[kind][a].read_bytes()).hexdigest()
                digest_b = hashlib.md5(written[kind][b].read_bytes()).hexdigest()
                if digest_a == digest_b:
                    problems.append(f"{kind}: {a} and {b} are the same file")
    # The second tissue has to differ from the first as well, or the full
    # model would be combining a profile with a copy of itself.
    for split, paths in full_tissue_files.items():
        digests = [hashlib.md5(p.read_bytes()).hexdigest() for p in paths]
        if len(set(digests)) != len(digests):
            problems.append(f"full: {split}'s two tissues are the same file")

    if problems:
        raise DataError("the splits are not distinct: " + "; ".join(problems))

    return ExampleData(out=out, files=written, unmatched=unmatched_written)
