"""Start from a model someone already fitted, instead of from scratch.

Useful when a new cohort is small. Fit on a large one, carry those weights
over, and continue on the small one, so the new run starts from something that
already knows what the counts look like.

**Version 0.1 had this and it did not do what it said.** Its
``set_parameter_requires_grad`` is documented as freezing and unfreezing layers
"based on specific conditions", and reads::

    if self.configs['type'] == 'simple':
        if any(keyword in name for keyword in ["omega", "theta", "pi_layer"]):
            param.requires_grad = True
        else:
            param.requires_grad = True

Both branches are the same. The condition is computed and cannot change the
outcome, so nothing was ever frozen. For the other two model kinds the body
says "IS NOT YET IMPLEMENTED" and sets everything trainable too. Finding 36.

So three things are different here:

- **Freezing freezes.** Naming a part that does not exist raises rather than
  quietly freezing nothing, which is the failure version 0.1 shipped.
- **The architectures have to match**, and a mismatch says which field differs
  and what the two values are. Version 0.1 called ``load_state_dict`` with no
  check, so a model fitted on different genes failed deep inside PyTorch, or
  worse, loaded.
- **It reports what it did**: how many parameter groups were loaded, skipped
  and frozen, so "I started from a pretrained model" is checkable afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from preffect.errors import CheckpointError

__all__ = ["PretrainedLoad", "load_pretrained", "freeze_parameters"]


@dataclass(frozen=True)
class PretrainedLoad:
    """What carrying weights over actually did."""

    source: Path
    written_by: str
    loaded: tuple[str, ...]
    #: Present in the file but not in this model, so not used.
    unused: tuple[str, ...]
    #: In this model but absent from the file, so left at their initial values.
    missing: tuple[str, ...]
    frozen: tuple[str, ...] = field(default=())

    @property
    def n_trainable(self) -> int:
        return len(self.loaded) + len(self.missing) - len(self.frozen)

    def __str__(self) -> str:
        parts = [
            f"loaded {len(self.loaded)} parameter group(s) from {self.source.name}",
            f"written by PREFFECT {self.written_by}",
        ]
        if self.missing:
            parts.append(f"{len(self.missing)} left at their initial values")
        if self.unused:
            parts.append(f"{len(self.unused)} in the file were not needed")
        if self.frozen:
            parts.append(f"{len(self.frozen)} frozen")
        return ", ".join(parts)


def freeze_parameters(model, parts) -> tuple[str, ...]:
    """Stop the named parts of a model from being trained.

    :param parts: name prefixes, for example ``("encoder",)`` to hold the
        encoder still while the decoder adapts to a new cohort.
    :raises CheckpointError: if a name matches nothing. Version 0.1's freezing
        silently matched nothing and trained everything, which looks exactly
        like freezing having worked.
    """
    parts = tuple(parts)
    if not parts:
        return ()

    names = [name for name, _ in model.named_parameters()]
    unmatched = [part for part in parts if not any(n.startswith(part) for n in names)]
    if unmatched:
        groups = sorted({n.split(".")[0] for n in names})
        raise CheckpointError(
            f"cannot freeze {', '.join(unmatched)}: no parameter starts with "
            f"that. This model's parts are: {', '.join(groups)}. Freezing "
            f"something that does not exist would train the whole model while "
            f"appearing to freeze part of it."
        )

    frozen = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(part) for part in parts):
            parameter.requires_grad = False
            frozen.append(name)
    return tuple(frozen)


def load_pretrained(
    model,
    path: str | Path,
    *,
    freeze=(),
    allow_missing: bool = False,
) -> PretrainedLoad:
    """Carry weights from a saved model into this one.

    :param model: the model to load into, already built.
    :param path: a checkpoint written by :func:`preffect.checkpoint.save_checkpoint`.
    :param freeze: name prefixes to hold still during the run that follows.
    :param allow_missing: proceed when the saved model does not cover every
        parameter here, leaving the rest at their initial values. Off by
        default, because the usual cause is loading the wrong file.
    :raises CheckpointError: if the architectures do not match, or if
        something is missing and ``allow_missing`` is not set.
    """
    from preffect.checkpoint import load_checkpoint

    path = Path(path)
    checkpoint = load_checkpoint(path)

    n_genes = getattr(model, "n_genes", None)
    if n_genes is not None and n_genes != checkpoint.n_genes:
        raise CheckpointError(
            f"the saved model was fitted on {checkpoint.n_genes} genes and this "
            f"one has {n_genes}. Weights cannot be carried between different "
            f"gene sets: column 7 does not mean the same gene in both. Fit the "
            f"new cohort on the same genes, or start from scratch."
        )

    saved_kind = checkpoint.model_config.kind
    this_kind = model.config.kind
    if saved_kind is not this_kind:
        raise CheckpointError(
            f"the saved model is a '{saved_kind.value}' model and this one is "
            f"'{this_kind.value}'. They do not have the same parts, so there is "
            f"nothing sensible to carry over."
        )

    for setting in ("latent_dim", "hidden_dim"):
        saved = getattr(checkpoint.model_config, setting)
        here = getattr(model.config, setting)
        if saved != here:
            raise CheckpointError(
                f"the saved model has {setting}={saved} and this one has "
                f"{here}. Every layer either side of it is a different shape, "
                f"so the weights do not fit."
            )

    here_shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    saved_state = checkpoint.model_state

    wrong_shape = [
        f"{name} is {tuple(saved_state[name].shape)} in the file and "
        f"{shape} here"
        for name, shape in here_shapes.items()
        if name in saved_state and tuple(saved_state[name].shape) != shape
    ]
    if wrong_shape:
        raise CheckpointError(
            "the saved weights do not fit this model: " + "; ".join(wrong_shape[:3])
        )

    loaded = tuple(sorted(set(here_shapes) & set(saved_state)))
    missing = tuple(sorted(set(here_shapes) - set(saved_state)))
    unused = tuple(sorted(set(saved_state) - set(here_shapes)))

    if missing and not allow_missing:
        raise CheckpointError(
            f"the saved model does not cover {len(missing)} parameter group(s) "
            f"here, starting with {', '.join(missing[:3])}. That usually means "
            f"the wrong file. Pass allow_missing=True to start those from "
            f"their initial values on purpose."
        )

    model.load_state_dict(saved_state, strict=False)
    frozen = freeze_parameters(model, freeze)

    return PretrainedLoad(
        source=path,
        written_by=checkpoint.written_by,
        loaded=loaded,
        unused=unused,
        missing=missing,
        frozen=frozen,
    )


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    """The parameters an optimiser should be given after freezing.

    Handing frozen parameters to an optimiser is harmless with Adam but wastes
    memory on their moment estimates, and it makes the optimiser's own record
    of what it is training disagree with the model's.
    """
    return [p for p in model.parameters() if p.requires_grad]
