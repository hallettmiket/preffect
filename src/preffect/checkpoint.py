"""Saving and reloading a fitted model.

A checkpoint holds **tensors and settings**, never a pickled Python object.

Version 0.1 saved a session with ``torch.save(self)``, pickling the whole live
object: the model, the optimiser, the datasets and the settings together.
PyTorch 2.6 changed ``torch.load`` to refuse pickled objects by default, so
every session version 0.1 ever wrote becomes unreadable the moment anyone
upgrades. The warning announcing that change was suppressed in version 0.1
rather than acted on.

Pickling the live object had two further costs. The file carried the training
data, so a checkpoint of a small model on a large cohort was enormous and
quietly contained the cohort. And it carried whatever layers the model had
allocated and never used, which for a simple model was 94% of the encoder.

Here a checkpoint is a plain dictionary of tensors and plain values. It loads
under PyTorch's safe default, it does not contain anybody's data, and it can be
read by a version of the code that is not the one that wrote it, because the
settings travel with it and are checked on the way back in.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import nn

from preffect import __version__
from preffect.config import CorrectionVariable, ModelConfig, PreffectConfig
from preffect.errors import CheckpointError

__all__ = [
    "CHECKPOINT_FORMAT",
    "TRAINING_HISTORY_KEY",
    "Checkpoint",
    "TrainingRecord",
    "save_checkpoint",
    "load_checkpoint",
    "training_history_record",
]

#: Bumped when the layout of a checkpoint changes in a way that makes an older
#: one unreadable. Independent of the package version: most releases do not
#: change the format.
CHECKPOINT_FORMAT = 1

#: Where a run's own record of how it fitted sits inside ``extra``. ``extra``
#: is free-form, so adding this does not change the format version: a file
#: written before it was kept loads exactly as it did, with nothing under this
#: key, and anything reading it has to cope with that.
TRAINING_HISTORY_KEY = "training_history"


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    """How the run that produced a model went, as plain numbers and sentences.

    A saved model says what it is but not how it came to be, and the two
    questions a reader asks of a fit are whether it had finished moving and
    whether it was stopped early. This carries the answer: the loss at every
    epoch for each split, when it stopped and why, and how long it took.

    **It is not the whole training history.** The per-epoch diagnostics that
    :class:`~preffect.training.TrainingHistory` collects, such as the gradient
    norms and whether the latent space collapsed, are left out. Keeping a few
    of its fields and defaulting the rest would hand a reader numbers that
    nothing measured.
    """

    #: Epoch numbers, starting at one, one per entry of the losses below.
    epochs: tuple[int, ...]
    #: Mean loss per batch over the training split, one per epoch.
    train_loss: tuple[float, ...]
    #: The same over the validation split, or ``None`` if none was given.
    validation_loss: tuple[float, ...] | None
    #: Epochs actually run, and epochs asked for. They differ after an early stop.
    n_epochs: int
    epochs_asked_for: int
    #: True when early stopping ended the run before its last epoch.
    stopped_early: bool
    #: The epoch whose weights were saved, which after an early stop is the
    #: best one rather than the last.
    best_epoch: int | None
    #: What early stopping watched: ``"loss"`` or ``"imputation"``.
    watched: str
    #: True when early stopping had to watch the training loss for want of a
    #: validation split, which is a weaker thing to stop on.
    watched_training_loss: bool
    #: One sentence saying how the run ended, ready to show to a reader.
    stop_reason: str
    #: Wall-clock seconds spent fitting, excluding reading the data.
    wall_seconds: float

    @classmethod
    def from_extra(cls, extra: dict[str, Any] | None) -> TrainingRecord | None:
        """Read the record out of a checkpoint's ``extra``, or ``None``.

        ``None`` means the model was saved before its history was kept, which
        is an ordinary thing for an older file and not an error.
        """
        stored = (extra or {}).get(TRAINING_HISTORY_KEY)
        if not isinstance(stored, dict) or "train_loss" not in stored:
            return None
        validation = stored.get("validation_loss")
        return cls(
            epochs=tuple(int(e) for e in stored.get("epochs", ())),
            train_loss=tuple(float(v) for v in stored["train_loss"]),
            validation_loss=(
                None if validation is None else tuple(float(v) for v in validation)
            ),
            n_epochs=int(stored.get("n_epochs", len(stored["train_loss"]))),
            epochs_asked_for=int(stored.get("epochs_asked_for", 0)),
            stopped_early=bool(stored.get("stopped_early", False)),
            best_epoch=(
                None if stored.get("best_epoch") is None else int(stored["best_epoch"])
            ),
            watched=str(stored.get("watched", "loss")),
            watched_training_loss=bool(stored.get("watched_training_loss", False)),
            stop_reason=str(stored.get("stop_reason", "")),
            wall_seconds=float(stored.get("wall_seconds", 0.0)),
        )


def _watched_quantity(history) -> str:
    """What early stopping was looking at, named for a reader."""
    if history.watched == "imputation":
        return "imputation error on the hidden entries"
    return "training loss" if history.watched_training_loss else "validation loss"


def _stop_reason(history, epochs_asked_for: int, patience: int | None) -> str:
    """One sentence saying how a run ended, and which weights it left behind."""
    run = len(history.epochs)
    if history.stopped_early:
        return (
            f"stopped early at epoch {run} of the {epochs_asked_for} asked for: "
            f"the {_watched_quantity(history)} had not improved for {patience} "
            f"epoch(s) in a row. The weights saved are from epoch "
            f"{history.best_epoch}, the best one."
        )
    if patience is not None:
        return (
            f"ran all {run} epoch(s) asked for: the "
            f"{_watched_quantity(history)} never went {patience} epoch(s) without "
            f"improving, so early stopping did not fire."
        )
    return (
        f"ran all {run} epoch(s) asked for. Nothing was watching for an early "
        f"stop, so the weights saved are the last epoch's."
    )


def training_history_record(
    history,
    *,
    epochs_asked_for: int,
    wall_seconds: float,
    patience: int | None = None,
) -> dict[str, Any]:
    """Reduce a training history to the plain values a checkpoint carries.

    Every value here is a number, a string or a list of them, so it survives
    PyTorch's safe loader and comes back without unpickling anything. Hand the
    result to :func:`save_checkpoint` under :data:`TRAINING_HISTORY_KEY` and
    read it back with :meth:`TrainingRecord.from_extra`.

    :param history: a :class:`~preffect.training.TrainingHistory`.
    :param epochs_asked_for: the epoch count the run was given, which after an
        early stop is larger than the number run.
    :param wall_seconds: how long the fit took, excluding reading the data.
    :param patience: the early-stopping patience in force, or ``None``.
    """
    validation = [
        float(record.validation.mean_loss)
        for record in history.epochs
        if record.validation is not None
    ]
    return {
        "epochs": [int(record.epoch) for record in history.epochs],
        "train_loss": [float(record.train.mean_loss) for record in history.epochs],
        "validation_loss": validation if validation else None,
        "n_epochs": len(history.epochs),
        "epochs_asked_for": int(epochs_asked_for),
        "stopped_early": bool(history.stopped_early),
        "best_epoch": history.best_epoch,
        "watched": str(history.watched),
        "watched_training_loss": bool(history.watched_training_loss),
        "stop_reason": _stop_reason(history, epochs_asked_for, patience),
        "wall_seconds": float(wall_seconds),
    }


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """What was read back from a file."""

    model_state: dict[str, torch.Tensor]
    model_config: ModelConfig
    n_genes: int
    gene_names: tuple[str, ...] | None
    optimiser_state: dict[str, Any] | None
    written_by: str
    format_version: int
    extra: dict[str, Any]
    #: Every setting the run used, as plain values, or ``None`` for a file
    #: written without them. Provenance: paths in it are the training
    #: machine's and are never used to find anything on reloading.
    full_config: dict[str, Any] | None = None

    @property
    def training(self) -> TrainingRecord | None:
        """How the run that wrote this file went, or ``None`` if it was not kept."""
        return TrainingRecord.from_extra(self.extra)

    def build_model(self):
        """Reconstruct the model and load its weights.

        The kind travels in the settings and the tissue count beside them, so
        a single or full model comes back as itself. This built a simple model
        whatever the file said until 2026-11-02, which for a graph model meant
        a PyTorch error about unexpected keys rather than a sentence, and
        nothing had reloaded a full model to find out (finding 55).
        """
        from preffect.errors import CheckpointError
        from preffect.workflow import build_model

        n_tissues = int(self.extra.get("n_tissues", 1))
        if self.model_config.kind.value == "full" and n_tissues < 2:
            raise CheckpointError(
                f"this checkpoint says it holds a full model but records "
                f"{n_tissues} tissue(s); the full model combines at least 2. The "
                f"file was written without its tissue count."
            )
        model = build_model(
            self.model_config.kind, self.n_genes, self.model_config, n_tissues
        )
        model.load_state_dict(self.model_state)
        return model


def _plain(value: Any) -> Any:
    """Reduce settings to things that survive a safe load.

    PyTorch's safe loader accepts tensors and a short list of built-in types,
    and refuses everything else, including our own enum classes. So enums go in
    as their string values and are turned back on the way out, which is also
    why :class:`~preffect.config.ModelConfig` accepts strings.
    """
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_config: ModelConfig,
    n_genes: int,
    gene_names: tuple[str, ...] | None = None,
    optimiser: torch.optim.Optimizer | None = None,
    full_config: PreffectConfig | None = None,
    **extra: Any,
) -> Path:
    """Write a model to disk.

    :param gene_names: the genes the model was fitted on, in order. Strongly
        worth supplying: without them nothing can check that the data a loaded
        model is applied to has its columns in the same order, which is the
        mistake version 0.1 made within a single run.
    :param full_config: recorded for the provenance of the run. Paths in it are
        stored as text and are not used on loading.
    """
    path = Path(path)
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "written_by": __version__,
        "model_state": model.state_dict(),
        "model_config": _plain(model_config),
        "n_genes": n_genes,
        "gene_names": list(gene_names) if gene_names is not None else None,
        "optimiser_state": optimiser.state_dict() if optimiser is not None else None,
        "full_config": _plain(full_config) if full_config is not None else None,
        "extra": _plain(extra),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read a model back.

    Loads with PyTorch's safe default, which refuses arbitrary pickled objects.

    :raises CheckpointError: if the file is missing, was written by an
        incompatible format, or is one of version 0.1's pickled sessions.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - the cause is reported verbatim
        raise CheckpointError(
            f"could not read {path}: {exc}. If this file was written by PREFFECT "
            f"0.1, it is a pickled Python object rather than a set of tensors, and "
            f"current PyTorch will not load it. Retrain, or load it with PREFFECT "
            f"0.1 under its own environment and export the weights."
        ) from exc

    if not isinstance(payload, dict) or "format_version" not in payload:
        raise CheckpointError(
            f"{path} is not a PREFFECT checkpoint: it has no format version."
        )

    found = payload["format_version"]
    if found != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"{path} is checkpoint format {found} and this is PREFFECT "
            f"{__version__}, which reads format {CHECKPOINT_FORMAT}."
        )

    config_values = dict(payload["model_config"])
    config_values["correction_variables"] = tuple(
        CorrectionVariable(**v) for v in config_values.get("correction_variables", [])
    )
    # ModelConfig coerces the enum fields back from their stored strings.
    model_config = ModelConfig(**config_values)

    gene_names = payload.get("gene_names")
    full_config = payload.get("full_config")
    return Checkpoint(
        model_state=payload["model_state"],
        model_config=model_config,
        n_genes=int(payload["n_genes"]),
        gene_names=tuple(gene_names) if gene_names is not None else None,
        optimiser_state=payload.get("optimiser_state"),
        written_by=payload.get("written_by", "unknown"),
        format_version=found,
        extra=payload.get("extra", {}),
        full_config=dict(full_config) if isinstance(full_config, dict) else None,
    )
