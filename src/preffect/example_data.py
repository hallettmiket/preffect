"""A small simulated cohort to try PREFFECT on, written by ``preffect example-data``.

Three splits with no sample in common, for each of the three models: the
simple model, the single model (the same counts plus a sample-sample network
with real edges in it) and the full model (two tissues per patient, under
different names, with the correspondence written down). About two seconds,
200 genes, three groups of samples that each raise a different block of
genes, so there is something real to find and you can tell whether the model
found it. The counts are negative binomial, which is what PREFFECT models.

**Version 0.1's vignettes ship a ``test/`` folder byte-identical to
``validation/``**, in all three of simple, single and full, and a published
figure shows a user pointing inference at that folder while the caption calls
it a held-out test set. The data here is generated rather than committed, for
two reasons: counts do not belong in a git repository, and a generator can
promise what a committed file cannot. The three splits are drawn from disjoint
samples by construction, and :func:`write_example_data` checks that and raises
rather than write them if they are not. Issue #11.

The full model's two files per split deliberately **do not share sample
names**, as the project's own data does not, so the correspondence goes in a
table under ``obsm['sample_association']`` with one column per tissue, which
is where :func:`preffect.data.tissues.align_tissues` looks. ``unmatched`` takes
the second tissue away from a share of patients, which is the shape of a real
cohort where some patients have no matched normal, and is what the full
model's padding exists for. It is zero by default and draws no random numbers
when it is zero, so the files written without it are byte for byte the files
written before the option existed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preffect.errors import ConfigError, DataError

__all__ = [
    "N_GENES",
    "N_SAMPLES",
    "N_SUBTYPES",
    "ExampleData",
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


def simulate(
    n_samples: int,
    n_genes: int,
    rng: np.random.Generator,
    offset: int,
    groups: np.ndarray | None = None,
    lift: float = 4.0,
):
    """Counts with a subtype signal in them, so an example has something to find.

    Each sample belongs to one of three groups, and each group raises a
    different block of genes. Negative binomial, which is what PREFFECT models,
    so the example is not quietly fitting Gaussian data.

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
    :returns: the counts, the group of each sample, and the sample names.
    """
    if groups is None:
        groups = rng.integers(0, N_SUBTYPES, size=n_samples)
    base = rng.lognormal(mean=3.0, sigma=1.0, size=n_genes)

    means = np.tile(base, (n_samples, 1))
    block = n_genes // (N_SUBTYPES + 1)
    for group in range(N_SUBTYPES):
        rows = groups == group
        means[np.ix_(rows, np.arange(group * block, (group + 1) * block))] *= lift

    depth = rng.lognormal(mean=0.0, sigma=0.25, size=(n_samples, 1))
    means = means * depth

    dispersion = rng.uniform(2.0, 8.0, size=n_genes)
    probability = dispersion / (dispersion + means)
    counts = rng.negative_binomial(dispersion, probability).astype(np.float32)

    names = [f"SAMPLE_{offset + i:04d}" for i in range(n_samples)]
    return counts, groups, names


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
        written[kind] = {}
        for split, n in N_SAMPLES.items():
            counts, groups, names = simulate(n, N_GENES, rng, offset)
            offset = _advance(offset, n)  # so no sample name is ever reused across splits

            adata = ad.AnnData(counts)
            # Object dtype, as the package writes names: under pandas 3 a list
            # becomes a string array that anndata 0.11 refuses to write, and a
            # fresh install can resolve exactly that pairing. Finding 48.
            adata.obs_names = pd.Index(names, dtype=object)
            adata.var_names = pd.Index([f"GENE_{j:04d}" for j in range(N_GENES)], dtype=object)
            adata.obs["subtype"] = pd.Categorical([f"group_{g}" for g in groups])
            adata.obs["batch"] = pd.Categorical(["A" if i % 2 else "B" for i in range(n)])
            if kind == "single":
                adata.obsm["sample_sample_adj"] = ring_network(n)

            folder = out / kind / split
            folder.mkdir(parents=True, exist_ok=True)
            # The loader looks for '*tau_<n>.h5ad', where n is the tissue
            # number. An example with one tissue is tau_1.
            path = folder / "counts.tau_1.h5ad"
            adata.write_h5ad(path)
            written[kind][split] = path
            sample_names[f"{kind}/{split}"] = names

    # The full model: two tissues per split, the same patients in each, and a
    # table saying which row of one is which row of the other. Written after
    # the others so the sample names carry on rather than repeating theirs.
    written["full"] = {}
    full_tissue_files: dict[str, list[Path]] = {}
    unmatched_written: dict[str, int] = {}
    for split, n in N_SAMPLES.items():
        first_counts, groups, first_names = simulate(n, N_GENES, rng, offset)
        offset = _advance(offset, n)
        # The same patients, so the same groups; a weaker block lift, so the
        # second tissue informs the first rather than repeating it.
        second_counts, _, second_names = simulate(n, N_GENES, rng, offset, groups=groups, lift=1.8)
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
        association = pd.DataFrame({"tau_1": first_names, "tau_2": second_names})
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
        for tissue, (tissue_counts, tissue_names, rows) in enumerate(
            ((first_counts, first_names, kept[0]), (second_counts, second_names, kept[1])),
            start=1,
        ):
            own_names = [name for name, keep in zip(tissue_names, rows, strict=True) if keep]
            own_groups = groups[rows]
            m = len(own_names)
            adata = ad.AnnData(tissue_counts if rows.all() else tissue_counts[rows])
            adata.obs_names = pd.Index(own_names, dtype=object)
            adata.var_names = pd.Index([f"GENE_{j:04d}" for j in range(N_GENES)], dtype=object)
            adata.obs["subtype"] = pd.Categorical([f"group_{g}" for g in own_groups])
            adata.obs["batch"] = pd.Categorical(["A" if i % 2 else "B" for i in range(m)])
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
