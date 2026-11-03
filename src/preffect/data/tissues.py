"""Matching samples across tissues, for the full model.

The full model takes several profiles from the same patient, for example a
tumour and the normal tissue beside it, and uses them together. To do that it
has to know which row of one file corresponds to which row of another.

**The files do not share sample names, deliberately.** In the project's own
data the names are ``tau1_0`` and ``tau2_0``, so a join on names would match
nothing. The correspondence is written down instead, in a table under
``obsm['sample_association']`` with one column per tissue, giving each sample's
counterpart in every other tissue. A patient with no counterpart in some tissue
has an empty cell in that tissue's column.

## A patient missing a tissue is padded, not refused

A cohort where some patients lack a matched normal is the common case in
practice, and until 2026-11-02 version 0.2 refused one outright. It now does
what version 0.1 did, ported rule for rule and then measured:

- **Every empty cell becomes a row of its own in that tissue**, called a ghost
  row here and in version 0.1. It gets a name of its own, every annotation
  missing, and its counts set to the rounded per-gene mean of that tissue's
  real samples.
- **The tissue's network is enlarged to cover the ghost rows, with no links**,
  so a ghost attends to nothing and nothing attends to it.
- **Ghost rows are left out of the expression reconstruction term, the two
  library-size terms and the imputation scores, and kept in the latent and
  network terms.** That split is version 0.1's, made in code rather than
  stated: it commented out the lines that would have removed ghosts from the
  latent and network parameters. The decision record of 2026-11-02 names it as
  inherited and measures it.
- **Hidden entries are never drawn from a ghost row**, which
  :class:`MultiTissueData` refuses rather than trusts.

``align_tissues(..., unmatched="refuse")`` brings the old refusal back. That is
not a historical courtesy: dropping the patients who are missing a tissue is
the real alternative to padding, and it is the arm the measurement in
``exp/19_unmatched_samples_padded`` compares padding against.

## What a ghost row cannot carry

Its annotations are all missing, by the rule above, so a padded run has no
correction variable and no batch label for it. Correction variables are
refused by name when any value is missing
(:func:`~preffect.data.tensors.encode_corrections`), which means a padded run
and correction variables cannot be combined today. Fill the annotations in
yourself, or drop the incomplete patients, or leave the correction variables
out.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import torch

from preffect.data.graph import ADJACENCY_KEY
from preffect.data.tensors import TensorData
from preffect.errors import DataError

__all__ = [
    "ASSOCIATION_KEY",
    "GHOST_COLUMN",
    "MultiTissueData",
    "TissueAlignment",
    "align_tissues",
    "ghost_rows",
    "unmatched_counts",
]

#: Where version 0.1 records which sample matches which, and where 0.2 looks.
ASSOCIATION_KEY = "sample_association"

#: The sample annotation :func:`align_tissues` writes to say which rows it
#: padded in. True for a ghost row, False for a real sample. It travels with
#: the tissue, so a step that only receives the tissues can still tell them
#: apart; :func:`ghost_rows` reads it, and reports no ghosts when it is absent.
GHOST_COLUMN = "preffect_ghost"

#: What an empty cell may look like in a correspondence table besides a real
#: missing value. Version 0.1 read the string ``NA`` as missing, and the
#: project's own tables use it, so that reading is kept.
EMPTY_CELL = "NA"


def ghost_rows(adata: ad.AnnData) -> np.ndarray:
    """Which rows of this tissue were padded in, as a boolean per row.

    All False for a tissue that was never padded, including one that never
    went through :func:`align_tissues`, so a caller does not have to ask
    whether the annotation is there.
    """
    if GHOST_COLUMN not in adata.obs.columns:
        return np.zeros(adata.n_obs, dtype=bool)
    return np.asarray(adata.obs[GHOST_COLUMN].to_numpy(), dtype=bool)


@dataclass(frozen=True, slots=True)
class TissueAlignment:
    """Tissues whose rows line up, and which of those rows are padding.

    Returned rather than a bare list so that the padding is handed back as a
    value, not only written into the tissues. The two say the same thing:
    ``ghosts[i]`` is :func:`ghost_rows` of ``tissues[i]``.
    """

    #: The tissues, reordered so row *i* is the same patient in every one.
    tissues: list[ad.AnnData]
    #: One boolean array per tissue, True where the row was padded in.
    ghosts: list[np.ndarray]

    @property
    def n_ghosts(self) -> list[int]:
        """How many rows each tissue had padded in."""
        return [int(g.sum()) for g in self.ghosts]

    @property
    def any_ghosts(self) -> bool:
        return any(bool(g.any()) for g in self.ghosts)


@dataclass(frozen=True, slots=True)
class MultiTissueData:
    """Several tissues whose rows refer to the same patients, in the same order.

    ``tissues[0]`` is the target: the profile the full model predicts. The rest
    inform that prediction without being predicted themselves, which is what
    version 0.1 does and what the paper describes as stabilising a profile
    using matched tissues.

    ``ghosts`` records which rows were padded in for a patient missing that
    tissue, one boolean tensor per tissue. Leave it unset when the cohort is
    complete: the terms that leave ghost rows out are then skipped entirely
    rather than given an all-true selection, so a complete cohort trains bit
    for bit as it did before padding existed.
    """

    tissues: list[TensorData]
    #: One ``[samples]`` boolean tensor per tissue, True where the row is
    #: padding. None when no tissue has any, which is the ordinary case.
    ghosts: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if not self.tissues:
            raise DataError("a multi-tissue dataset needs at least one tissue")
        sizes = {t.n_samples for t in self.tissues}
        if len(sizes) > 1:
            raise DataError(
                f"the tissues have different sample counts {sorted(sizes)}; they "
                f"must be aligned first"
            )
        genes = {t.n_genes for t in self.tissues}
        if len(genes) > 1:
            raise DataError(f"the tissues have different gene counts {sorted(genes)}")

        if self.ghosts is None:
            return
        if len(self.ghosts) != len(self.tissues):
            raise DataError(
                f"there are {len(self.tissues)} tissue(s) but {len(self.ghosts)} "
                f"record(s) of which rows are padding; there must be one per tissue"
            )
        for index, (tissue, ghost) in enumerate(
            zip(self.tissues, self.ghosts, strict=True), start=1
        ):
            if ghost.dtype is not torch.bool or ghost.dim() != 1:
                raise DataError(
                    f"tissue {index}'s record of padded rows is {ghost.dtype} of "
                    f"shape {tuple(ghost.shape)}; it has to be one boolean per row"
                )
            if ghost.shape[0] != tissue.n_samples:
                raise DataError(
                    f"tissue {index} has {tissue.n_samples} rows but its record of "
                    f"padded rows covers {ghost.shape[0]}"
                )
            self._refuse_hidden_ghosts(index, tissue, ghost)

    @staticmethod
    def _refuse_hidden_ghosts(index: int, tissue: TensorData, ghost: torch.Tensor) -> None:
        """A hidden entry in a padded row would score the model on a value nobody measured.

        A ghost row's counts are the per-gene mean of the real samples, so
        hiding one of them and asking the model to recover it scores it against
        something this package made up. Version 0.1 kept ghost rows out of the
        masked-entry bookkeeping in its batch iterator; this refuses the mask
        instead, because a mask that hides a ghost entry was built wrongly and
        quietly dropping those entries would leave the caller's count of hidden
        entries disagreeing with what was scored.
        """
        if tissue.mask is None or not bool(ghost.any()):
            return
        rows = ghost.detach().cpu().numpy()
        hidden_in_ghosts = int(tissue.mask.hidden[rows].sum())
        if hidden_in_ghosts:
            where = "the target tissue" if index == 1 else f"tissue {index}"
            raise DataError(
                f"the mask for {where} hides {hidden_in_ghosts} entr(ies) in rows "
                f"that were padded in for patients missing that tissue. Those "
                f"counts are the per-gene mean of the real samples, so recovering "
                f"them measures nothing. Build the mask over the real rows and "
                f"leave the padded ones visible."
            )

    @property
    def n_tissues(self) -> int:
        return len(self.tissues)

    @property
    def n_samples(self) -> int:
        return self.tissues[0].n_samples

    @property
    def n_genes(self) -> int:
        return self.tissues[0].n_genes

    @property
    def target(self) -> TensorData:
        """The tissue whose counts the model predicts."""
        return self.tissues[0]

    @property
    def n_ghosts(self) -> list[int]:
        """How many rows of each tissue are padding."""
        if self.ghosts is None:
            return [0] * self.n_tissues
        return [int(g.sum()) for g in self.ghosts]

    def target_real_rows(self) -> torch.Tensor | None:
        """Which rows of the target tissue are real samples, or None if all are.

        None rather than an all-true tensor on purpose. The objective skips the
        selection entirely when this is None, so a cohort with nothing padded
        gives the same numbers to the last bit as it did before padding
        existed.
        """
        if self.ghosts is None or not bool(self.ghosts[0].any()):
            return None
        return ~self.ghosts[0]

    def select(self, positions: np.ndarray | torch.Tensor) -> MultiTissueData:
        """One mini-batch, taken from every tissue at the same positions.

        The same positions in every tissue, which is only meaningful because
        the rows were aligned first. The record of which rows are padding is
        taken at the same positions, so a batch knows which of its own rows
        are padding without having to look back at the whole split.
        """
        tissues = [t.select(positions) for t in self.tissues]
        if self.ghosts is None:
            return MultiTissueData(tissues)
        ghosts = [
            g[torch.as_tensor(positions, dtype=torch.long, device=g.device)]
            for g in self.ghosts
        ]
        return MultiTissueData(tissues, ghosts=ghosts)

    def to(self, device: str | torch.device) -> MultiTissueData:
        """Return a copy with every tissue's tensors moved to ``device``.

        Every tissue goes to the same device. The full model's terms are
        summed across tissues, so a split set would fail at the first sum, and
        failing here with a clear message beats failing there with an obscure
        one. The record of which rows are padding moves too: it selects rows of
        tensors that have moved.
        """
        return MultiTissueData(
            [t.to(device) for t in self.tissues],
            ghosts=None if self.ghosts is None else [g.to(device) for g in self.ghosts],
        )


def _combined_table(
    adatas: list[ad.AnnData], expected: list[str], split: str
) -> pd.DataFrame:
    """One row per patient, built from every tissue's copy of the table.

    A tissue's ``obsm`` frame has one row per sample of that tissue, so a
    patient who is missing tissue 1 cannot appear in tissue 1's copy at all.
    Reading only the first tissue's copy would therefore never see them.
    Version 0.1 concatenated every copy and dropped the repeats, and so does
    this; on a complete cohort, where every copy is the same table, the result
    is that table.
    """
    frames = []
    for index, adata in enumerate(adatas, start=1):
        if ASSOCIATION_KEY not in adata.obsm:
            if index == 1:
                raise DataError(
                    f"{split} tissue 1 has no sample correspondence table under "
                    f"obsm[{ASSOCIATION_KEY!r}], which the full model needs to know "
                    f"which row of one tissue matches which row of another. Present: "
                    f"{sorted(adata.obsm) or '(nothing)'}"
                )
            continue
        table = adata.obsm[ASSOCIATION_KEY]
        missing_columns = [c for c in expected if c not in table.columns]
        if missing_columns:
            where = (
                f"{split}'s correspondence table"
                if index == 1
                else f"{split} tissue {index}'s correspondence table"
            )
            raise DataError(
                f"{where} is missing column(s) {missing_columns} for "
                f"{len(adatas)} tissues. It has: {list(table.columns)}"
            )
        # The index is that tissue's own sample names, which differ from tissue
        # to tissue and would stop two copies of one row being recognised as
        # the same row. Dropped here for that reason.
        frames.append(
            pd.DataFrame({c: list(table[c]) for c in expected}).replace(
                {EMPTY_CELL: np.nan}
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(ignore_index=True)


def unmatched_counts(adatas: list[ad.AnnData], split: str = "data") -> dict[str, int]:
    """How many patients have no sample in each tissue.

    What ``preffect check`` reports, and what decides whether padding has
    anything to do. An empty dictionary when there is only one tissue, or when
    the first tissue carries no correspondence table, since neither case has a
    count to give.
    """
    if len(adatas) < 2 or ASSOCIATION_KEY not in adatas[0].obsm:
        return {}
    expected = [f"tau_{i}" for i in range(1, len(adatas) + 1)]
    table = _combined_table(adatas, expected, split)
    return {c: int(table[c].isna().sum()) for c in expected}


def _ghost_names(column: str, how_many: int, taken: set[str], split: str) -> list[str]:
    """Names for the rows padded into one tissue, as version 0.1 named them.

    ``tau_2_1``, ``tau_2_2`` and so on for the second tissue. Version 0.1 did
    not check these against the names already in use; this refuses a collision
    rather than producing two rows with one name, because a repeated sample
    name silently drops a sample (finding 62).
    """
    names = [f"{column}_{k}" for k in range(1, how_many + 1)]
    clashing = [n for n in names if n in taken]
    if clashing:
        raise DataError(
            f"{split} cannot pad {column}: the name(s) {clashing[:5]} are already "
            f"in use by real samples, and version 0.1's naming scheme for padded "
            f"rows would repeat them. Rename those samples."
        )
    return names


def _check_column(
    column: str, values: list, adata: ad.AnnData, index: int, split: str
) -> None:
    """Refuse a correspondence column that does not name this tissue's samples once each.

    Two failures, both of which used to pass silently and both of which lose a
    sample. A name the tissue does not have was already refused; a name given
    twice made ``adata[wanted]`` repeat one row and drop another, and a sample
    the column never names was dropped outright. Finding 62.
    """
    present = set(map(str, adata.obs_names))
    absent = [name for name in values if name not in present]
    if absent:
        shown = ", ".join(map(str, absent[:5]))
        raise DataError(
            f"{split} tissue {index} is missing {len(absent)} sample(s) its "
            f"correspondence table names: {shown}"
            + (f" and {len(absent) - 5} more" if len(absent) > 5 else "")
        )

    seen: dict[str, int] = {}
    for name in values:
        seen[name] = seen.get(name, 0) + 1
    repeated = sorted(n for n, count in seen.items() if count > 1)
    if repeated:
        shown = ", ".join(repeated[:5])
        raise DataError(
            f"{split}'s correspondence table names {len(repeated)} sample(s) of "
            f"tissue {index} as the counterpart of more than one patient: {shown}"
            + (f" and {len(repeated) - 5} more" if len(repeated) > 5 else "")
            + ". A sample belongs to one patient, and aligning on a table like "
            "this repeats one row and drops another without saying so."
        )

    never_named = sorted(present - set(map(str, values)))
    if never_named:
        shown = ", ".join(never_named[:5])
        raise DataError(
            f"{split} tissue {index} has {len(never_named)} sample(s) the "
            f"correspondence table never names: {shown}"
            + (f" and {len(never_named) - 5} more" if len(never_named) > 5 else "")
            + ". Aligning would drop them without saying so. Add them to the "
            "table, or take them out of the file."
        )


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)


def ghost_profile(counts: np.ndarray) -> np.ndarray:
    """The counts a padded row carries: the rounded per-gene mean of the real samples.

    Version 0.1's rule, read from its code rather than its comment, which says
    the row is zero "unless ave_ghost_expr is true" and names a setting that
    does not exist. Genes that are missing in every sample average to nothing
    and are filled with zero, as version 0.1 did.
    """
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(counts, axis=0) if counts.shape[0] else np.zeros(counts.shape[1])
    return np.round(np.nan_to_num(mean)).astype(np.float32)


def _pad_tissue(
    adata: ad.AnnData,
    names: list[str],
    ghost: np.ndarray,
    table: pd.DataFrame,
) -> ad.AnnData:
    """One tissue reordered onto the table's rows, with its padded rows filled in.

    A tissue with nothing to pad is reordered by subsetting, exactly as it was
    before padding existed, so a complete cohort reaches the model in the same
    form it always did, sparse matrices and all. Only a tissue that is actually
    short is rebuilt.

    Rebuilding carries everything the file held: the gene annotations, anything
    under ``varm`` and ``uns``, every layer, and every ``obsm`` entry. The
    network gets the ghost rows with no links; any other ``obsm`` array gets
    zeros there, and any ``obsm`` frame gets missing values, matching the rule
    for the annotations.
    """
    index = pd.Index(names, dtype=object)
    own = table.copy()
    own.index = index

    if not ghost.any():
        subset = adata[names].copy()
        subset.obs[GHOST_COLUMN] = ghost
        subset.obsm[ASSOCIATION_KEY] = own
        return subset

    positions = {str(n): i for i, n in enumerate(adata.obs_names)}
    real = np.flatnonzero(~ghost)
    source = np.array([positions[names[i]] for i in real], dtype=np.int64)
    n_rows = len(names)

    def rebuild(values: np.ndarray, fill) -> np.ndarray:
        out = np.empty((n_rows, values.shape[1]), dtype=values.dtype)
        out[real] = values[source]
        out[ghost] = fill
        return out

    counts = _dense(adata.X)
    padded = ad.AnnData(
        X=rebuild(counts, ghost_profile(counts)),
        obs=adata.obs.reindex(names).set_axis(index),
        var=adata.var.copy(),
    )
    padded.obs[GHOST_COLUMN] = ghost
    if len(adata.varm):
        padded.varm = adata.varm
    padded.uns = dict(adata.uns)

    for key, layer in adata.layers.items():
        # anndata reports the counts themselves under a layer named None, and
        # writing a layer back under that name replaces the counts. Doing so
        # here silently overwrote every padded row with an unrounded gene mean
        # until the test for the rounded profile caught it.
        if key is None:
            continue
        dense = _dense(layer)
        fill = np.nan_to_num(np.nanmean(dense.astype(np.float64), axis=0))
        padded.layers[key] = rebuild(dense, fill.astype(dense.dtype, copy=False))

    for key, value in adata.obsm.items():
        if key == ASSOCIATION_KEY:
            continue
        if isinstance(value, pd.DataFrame):
            padded.obsm[key] = value.reindex(names).set_axis(index)
        elif key == ADJACENCY_KEY:
            padded.obsm[key] = _pad_network(_dense(value), source, real, n_rows)
        else:
            padded.obsm[key] = rebuild(_dense(value), 0)

    padded.obsm[ASSOCIATION_KEY] = own
    return padded


def _pad_network(network: np.ndarray, source: np.ndarray, real: np.ndarray, n_rows: int):
    """The tissue's network on the new rows, with the padded ones linked to nothing.

    Square over the new row count, the real samples' links carried across to
    wherever those samples now sit, and nothing at all on a padded row. Version
    0.1 did the same in two steps, reordering the stored matrix and then
    padding it with zeros on the right and the bottom.
    """
    out = np.zeros((n_rows, n_rows), dtype=network.dtype)
    out[np.ix_(real, real)] = network[np.ix_(source, source)]
    return out


def align_tissues(
    adatas: list[ad.AnnData],
    split: str = "data",
    unmatched: str = "pad",
) -> TissueAlignment:
    """Reorder every tissue so row *i* is the same patient throughout.

    Uses the correspondence table, combined across the tissues that carry it so
    that a patient missing the first tissue is still seen, and takes its row
    order as the order for all of them.

    :param unmatched: what to do about a patient with no sample in some tissue.
        ``"pad"``, the default, gives that tissue a row of its own for the
        patient, as described in this module's documentation. ``"refuse"``
        raises instead, which is what version 0.2 did until 2026-11-02 and
        what the caller wants in order to drop the incomplete patients and
        work with a complete cohort.
    :returns: the aligned tissues and, beside them, which of each tissue's rows
        are padding.
    :raises DataError: if the table is missing, names a tissue that is not
        there, does not name this tissue's samples once each, or leaves a
        patient unmatched while ``unmatched="refuse"``.
    """
    if unmatched not in ("pad", "refuse"):
        raise DataError(
            f"unmatched must be 'pad' or 'refuse', got {unmatched!r}. 'pad' gives "
            f"a patient missing a tissue a row of its own there; 'refuse' stops, "
            f"so the caller can drop the incomplete patients."
        )
    if len(adatas) < 2:
        return TissueAlignment(
            tissues=list(adatas),
            ghosts=[np.zeros(a.n_obs, dtype=bool) for a in adatas],
        )

    expected = [f"tau_{i}" for i in range(1, len(adatas) + 1)]
    table = _combined_table(adatas, expected, split)

    counts = {c: int(table[c].isna().sum()) for c in expected}
    if unmatched == "refuse" and any(counts.values()):
        raise DataError(
            f"{split} has samples with no counterpart in every tissue {counts}, "
            f"and this call asked for them to be refused. Drop the incomplete "
            f"patients, use the single model on one tissue, or leave the "
            f"default in place, which pads them."
        )

    taken = {str(n) for a in adatas for n in a.obs_names}
    taken |= {str(v) for c in expected for v in table[c] if not pd.isna(v)}

    plans: list[tuple[ad.AnnData, list[str], np.ndarray]] = []
    ghosts: list[np.ndarray] = []
    filled = table.copy()
    for index, (adata, column) in enumerate(zip(adatas, expected, strict=True), start=1):
        ghost = filled[column].isna().to_numpy()
        names = [str(v) for v in filled[column]]
        if ghost.any():
            fresh = _ghost_names(column, int(ghost.sum()), taken, split)
            taken |= set(fresh)
            for position, name in zip(np.flatnonzero(ghost), fresh, strict=True):
                names[position] = name
        filled[column] = names
        _check_column(column, [n for n, g in zip(names, ghost, strict=True) if not g],
                      adata, index, split)
        ghosts.append(ghost)
        plans.append((adata, names, ghost))

    # Built in a second pass so that every column of the table carries its
    # padded names before any tissue stores a copy of it.
    tissues = [_pad_tissue(adata, names, ghost, filled) for adata, names, ghost in plans]
    return TissueAlignment(tissues=tissues, ghosts=ghosts)
