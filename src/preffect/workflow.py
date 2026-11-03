"""The steps the command line and the interface share.

Reading a split and checking it, turning checked tissues into what a model
takes, building a model of a given kind, and running a saved model over a
folder of counts. The command line (``preffect.cli``) and the graphical
interface (``preffect.gui``) both do these things, and a step that lived in
one of them would drift from the other. Nothing here prints: each caller
reports in its own way, and :func:`apply_checkpoint` takes a ``report``
callable for that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SplitReport",
    "TissueSummary",
    "apply_checkpoint",
    "build_model",
    "describe_network",
    "inspect_split",
    "read_split",
    "to_model_data",
]


def read_split(folder: Path, n_tissues: int, unmatched: str = "pad"):
    """Read a split, check it, and align the tissues if there are several.

    :param unmatched: what to do about a patient with no sample in some
        tissue, passed straight to
        :func:`~preffect.data.tissues.align_tissues`. The default pads a row
        in; ``"refuse"`` stops instead, so a caller can drop the incomplete
        patients and work with a complete cohort.
    :returns: the tissues. Which rows were padded in travels with them, in the
        sample annotation :data:`~preffect.data.tissues.GHOST_COLUMN`, which
        is what :func:`to_model_data` reads.
    :raises PreffectError: if the folder is missing, holds no count files, or
        the counts would stop a run.
    """
    from preffect.data.checks import check_tissues
    from preffect.data.loading import load_tissues
    from preffect.data.tissues import align_tissues

    tissues = load_tissues(folder, n_tissues)
    check_tissues(tissues, split=folder.name)
    if n_tissues <= 1:
        return tissues
    return align_tissues(tissues, split=folder.name, unmatched=unmatched).tissues


def to_model_data(tissues, model_config, with_network: bool):
    """Turn checked tissues into what a model takes: one bundle, or one per tissue.

    Several tissues come back as a bundle carrying which of their rows were
    padded in for a patient missing that tissue, read from the annotation
    :func:`~preffect.data.tissues.align_tissues` wrote. Nothing is carried
    when no row was padded, so a complete cohort reaches the objective in the
    form it always did.
    """
    import torch

    from preffect.data.tensors import to_tensors
    from preffect.data.tissues import MultiTissueData, ghost_rows

    parts = [
        to_tensors(t, model_config.correction_variables, with_network=with_network)
        for t in tissues
    ]
    if len(parts) == 1:
        return parts[0]
    ghosts = [ghost_rows(t) for t in tissues]
    if not any(g.any() for g in ghosts):
        return MultiTissueData(parts)
    # copy(), because an annotation read from a file is not writable and
    # torch.from_numpy on a read-only array warns about undefined behaviour.
    return MultiTissueData(parts, ghosts=[torch.from_numpy(g.copy()) for g in ghosts])


def hide_entries(data, fraction: float, seed: int | None):
    """Hide a fraction of the entries at random, keeping the full counts as the targets.

    What ``--hide`` does: the model is shown counts with the hidden entries set
    to zero and scored against the originals, so its error on the hidden
    entries can be reported and watched (``TrainConfig.stop_on``). With a seed
    the same entries are hidden every time.
    """
    from dataclasses import replace

    import numpy as np
    import torch

    from preffect.data.masking import apply_mask, mcar_mask
    from preffect.data.tensors import TensorData
    from preffect.errors import ConfigError

    if not isinstance(data, TensorData):
        raise ConfigError(
            "--hide works for the simple and single models; the full model's data "
            "has one table per tissue, and hiding entries across them is not built"
        )
    counts = data.inputs.detach().cpu().numpy()
    mask = mcar_mask(counts.shape[0], counts.shape[1], fraction, np.random.default_rng(seed))
    shown = torch.from_numpy(apply_mask(counts, mask)).to(data.inputs.dtype)
    # library_scale cleared on purpose, so it is derived from the mask being
    # applied here rather than copied from whatever the data arrived with. A
    # scale given by hand wins over the mask, which is what lets a mini-batch
    # carry one, and it would silently win here too.
    return replace(data, inputs=shown, mask=mask, library_scale=None)


def build_model(kind, n_genes: int, model_config, n_tissues: int):
    """An untrained model of the requested kind."""
    from preffect.config import ModelKind

    if kind is ModelKind.SIMPLE:
        from preffect.model import SimpleModel

        return SimpleModel(n_genes, model_config)
    if kind is ModelKind.SINGLE:
        from preffect.graph_model import SingleModel

        return SingleModel(n_genes, model_config)
    from preffect.full_model import FullModel

    return FullModel(n_genes, model_config, n_tissues=n_tissues)


def describe_network(tissue) -> str:
    """Say how many edges the network has, not merely that it exists.

    Saying "a sample-sample network" was misleading: every dataset version 0.1
    shipped carries an identity matrix there, which is a network with no edges
    at all. Someone reading that line would reasonably fit a graph model and
    get nothing for ten times the parameters. Findings 35 and 54.
    """
    import numpy as np

    from preffect.data.graph import ADJACENCY_KEY, has_adjacency

    if not has_adjacency(tissue):
        return "no network"

    matrix = tissue.obsm[ADJACENCY_KEY]
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    dense = np.array(dense)
    np.fill_diagonal(dense, 0)
    edges = int(np.count_nonzero(dense))

    if edges == 0:
        return (
            "a network with NO edges between samples, so the graph models "
            "have nothing to use; fit the simple model"
        )
    return f"a network with {edges // 2} edge(s), {edges / tissue.n_obs:.1f} per sample"


@dataclass(frozen=True)
class TissueSummary:
    """One tissue of a split, as the check reports it."""

    samples: int
    genes: int
    network: str
    #: How many patients in this split have no sample in this tissue. Those
    #: patients get a row of per-gene means padded in here when the full model
    #: reads the split; zero means nothing is padded. Always zero for a split
    #: with one tissue, which has no counterparts to be missing.
    unmatched: int = 0


@dataclass(frozen=True)
class SplitReport:
    """What :func:`inspect_split` found: the tissues, and what would stop a run.

    ``problem`` is the sentence a run would have refused with, or ``None`` if
    the counts look usable. The tissues are listed even when there is a
    problem, whenever they could be read at all, so a reader sees what was
    there.
    """

    folder: Path
    tissues: tuple[TissueSummary, ...]
    problem: str | None = None

    @property
    def usable(self) -> bool:
        """True if nothing would stop a run on this split."""
        return self.problem is None


def inspect_split(folder: Path, n_tissues: int = 1) -> SplitReport:
    """Read a split and report on it without fitting anything.

    The check the command line's ``check`` makes, as a value rather than as
    printed lines, so the interface can show the same thing. A folder that
    cannot be read at all gives a report with no tissues and the message.
    """
    from preffect.data.checks import check_tissues
    from preffect.data.loading import load_tissues
    from preffect.data.tissues import unmatched_counts
    from preffect.errors import PreffectError

    folder = Path(folder)
    try:
        tissues = load_tissues(folder, n_tissues)
    except PreffectError as error:
        return SplitReport(folder=folder, tissues=(), problem=str(error))
    try:
        unmatched = unmatched_counts(tissues, split=folder.name)
    except PreffectError:
        # A correspondence table that cannot be read at all is reported by
        # check_tissues or by the alignment, with its own message. Counting is
        # the lesser question, so it gives up quietly rather than pre-empting
        # the sentence that says what is actually wrong.
        unmatched = {}
    summaries = tuple(
        TissueSummary(
            samples=t.n_obs,
            genes=t.n_vars,
            network=describe_network(t),
            unmatched=unmatched.get(f"tau_{i}", 0),
        )
        for i, t in enumerate(tissues, start=1)
    )
    try:
        check_tissues(tissues, split=folder.name)
    except PreffectError as error:
        return SplitReport(folder=folder, tissues=summaries, problem=str(error))
    return SplitReport(folder=folder, tissues=summaries)


def apply_checkpoint(
    model_path: Path,
    data_folder: Path,
    *,
    n_tissues: int = 1,
    device="cpu",
    report: Callable[[str], None] | None = None,
):
    """Load a saved model, read a folder of counts, and run the model over it.

    :param n_tissues: how many tissue files the folder holds. Only consulted
        if the model did not record its own count; a model saved by this
        version always does.
    :param device: where to run, as returned by
        :func:`preffect.training.resolve_device`. A string is passed to
        PyTorch as it is; resolve it first if it might name a device the
        machine lacks.
    :param report: called with each progress line; leave unset for silence.
    :returns: the :class:`~preffect.inference.InferenceResult`, the tissues it
        was computed on with their genes already matched to the model's, and
        the checkpoint. Where the full model was given a cohort with patients
        missing the predicted tissue, the rows padded in for them are dropped
        from the result **and** from the first tissue returned, so the two
        still line up row for row.
    """
    from preffect.checkpoint import load_checkpoint
    from preffect.data.selection import GeneSelection, apply_gene_selection
    from preffect.data.tissues import ghost_rows
    from preffect.inference import infer

    def say(line: str) -> None:
        if report is not None:
            report(line)

    data_folder = Path(data_folder)
    checkpoint = load_checkpoint(model_path)
    say(
        f"loaded a {checkpoint.model_config.kind.value} model written by PREFFECT "
        f"{checkpoint.written_by}, applying on {device}"
    )

    n_tissues = int(checkpoint.extra.get("n_tissues", n_tissues))
    kind = checkpoint.model_config.kind
    model = build_model(kind, checkpoint.n_genes, checkpoint.model_config, n_tissues)
    model.load_state_dict(checkpoint.model_state)
    model.to(device)

    tissues = read_split(data_folder, n_tissues)
    if checkpoint.gene_names is not None:
        # Match the genes the model was fitted on, by name. Without this a
        # column could mean a different gene than it did during training,
        # which is finding 18.
        selection = GeneSelection(names=checkpoint.gene_names, basis="from the model")
        tissues = apply_gene_selection(tissues, selection, data_folder.name)
        say(f"matched {len(selection)} genes to the ones the model was fitted on")

    data = to_model_data(tissues, checkpoint.model_config, kind.uses_adjacency).to(device)
    result = infer(model, data)
    if result.n_padded_rows_dropped:
        say(
            f"dropped {result.n_padded_rows_dropped} row(s) that were padded in "
            f"for patients with no sample in the predicted tissue; nothing in "
            f"them was measured"
        )
        real = ~ghost_rows(tissues[0])
        tissues = [tissues[0][real].copy(), *tissues[1:]]
    return result, tissues, checkpoint
