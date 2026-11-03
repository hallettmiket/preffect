"""Fitting a model, and recording honestly what happened while it fitted.

## The number this module exists to get right

An epoch's loss is the **mean over its batches**. Version 0.1 computed the
training loss that way and the validation loss a different way: it divided the
accumulated validation loss by the number of *tissues*, which is one for the
simple and single models, so the figure it printed, plotted, sent to
TensorBoard and compared against for early stopping was a **sum presented as a
mean**.

That makes it incomparable across batch sizes, because the number of batches
changes with the batch size. On four of version 0.1's own run directories,
differing only in epochs and batch size, the reported values spanned 33% and
the corrected values 4.4%, and the two orderings disagreed.

Here both splits go through the same function, which divides by the number of
batches it actually ran, and that count is recorded alongside the loss so the
arithmetic can be checked after the fact rather than trusted.

## Early stopping

Off unless asked for. It watches the validation loss and stops when it has not
improved by more than a stated amount for a stated number of epochs in a row,
restoring the weights from the best epoch. With no validation data it watches
the training loss instead and says so, because stopping early on the training
loss is a different and weaker thing.
"""

from __future__ import annotations

import copy
import os
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn

from preffect.config import PreffectConfig, RuntimeConfig
from preffect.data.batching import batch_indices
from preffect.data.tensors import TensorData
from preffect.data.tissues import MultiTissueData
from preffect.errors import ConfigError
from preffect.loss import compute_loss

__all__ = [
    "EpochRecord",
    "TrainingHistory",
    "SplitResult",
    "configure_runtime",
    "run_epoch",
    "train",
]


def configure_runtime(config: PreffectConfig) -> None:
    """Apply the runtime settings to PyTorch, once, before building the model.

    Before building, not only before ``train``. The seed and the thread count
    both act at construction: the weights are drawn then, and the sums that
    draw them run at whatever thread count is in force, so a model built at
    one count and trained at another is not the model a later call in the
    same process would build (finding 26 for the seed, issue #23 for the
    threads). ``train`` calls this too, and warns if it finds the thread count
    changing under a model that already exists.

    **Set a thread count.** PyTorch defaults to one thread per core, which on a
    large shared machine means a hundred or more. For the tensor sizes PREFFECT
    works with, coordinating that many threads costs far more than the
    arithmetic they do. Measured on this project's own graph model, 24 training
    steps took:

    ===========  ========
    threads      time
    ===========  ========
    1            0.082 s
    4            0.082 s
    16           0.107 s
    112          0.200 s
    ===========  ========

    Median of five runs on a loaded shared machine, so about two and a half
    times slower at the default. An earlier version of this docstring said nine
    hundred times, from a single timing that does not reproduce; see finding 24
    for what went wrong with it. Version 0.1 never set the number either, though
    its vignette notebook does so by hand, which suggests the problem was known
    and worked around rather than fixed.

    The right number depends on the data. A few threads is right for small
    cohorts; more helps once the matrices are large. Leaving ``num_threads``
    unset leaves PyTorch alone, which on a shared machine is rarely what you
    want.
    """
    if config.runtime.num_threads is not None:
        torch.set_num_threads(config.runtime.num_threads)

    if config.runtime.seed is not None:
        torch.manual_seed(config.runtime.seed)

    if config.runtime.deterministic:
        # Bit-for-bit repeatability, measured on an A100; see the decision
        # record 2026-10-27_gpu_runs_are_deterministic_when_asked.md. The CPU
        # needs the seed alone. A GPU needs four more things: cuBLAS given one
        # fixed workspace, which PyTorch reads from the environment; cuDNN told
        # not to choose its algorithm by timing it; cuDNN's deterministic
        # kernels; and deterministic algorithms everywhere else, which is what
        # makes the graph attention's scatter-add exact. On the single model,
        # 38 of 38 parameter tensors differed between two seeded runs without
        # these and 0 of 38 with them. Setting the workspace variable after
        # CUDA had already been used in the process also worked, measured,
        # though PyTorch's documentation asks for it before launch. setdefault,
        # so that a value chosen in the environment wins.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only: an operation with no deterministic kernel warns, naming
        # itself, rather than stopping the run. That is the report issue #9
        # asked for. No operation PREFFECT uses warned.
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Refuse a device the machine lacks now, before any data is read. The
    # command line calls this first for exactly that reason.
    resolve_device(config.runtime)


def resolve_device(runtime: RuntimeConfig) -> torch.device:
    """The device a run will use, refused by name if the machine lacks it.

    Called by ``train`` and by ``apply`` before any data is read. A run that
    asks for a GPU on a machine without one, or for ``cuda:3`` on a machine
    with two cards, would otherwise fail inside PyTorch after the loading, or,
    before finding 45, not fail at all: the string was checked and nothing was
    ever moved, so ``--device cuda`` trained on the CPU without a word.
    """
    if not runtime.uses_cuda:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ConfigError(
            f"device is {runtime.device!r} but this machine has no usable CUDA "
            f"device. Use device='cpu', or run where there is a GPU."
        )
    device = torch.device(runtime.device)
    count = torch.cuda.device_count()
    index = device.index if device.index is not None else 0
    if index >= count:
        raise ConfigError(
            f"device is {runtime.device!r} but this machine has {count} CUDA "
            f"device(s), numbered 0 to {count - 1}."
        )
    return device


@dataclass(frozen=True, slots=True)
class SplitResult:
    """What one pass over one split produced."""

    mean_loss: float
    mean_terms: dict[str, float]
    n_batches: int
    n_samples: int

    #: Latent dimensions whose mean actually varies between samples. A latent
    #: space that has collapsed carries no information about which sample is
    #: which, and this is how you see it happen rather than infer it afterwards
    #: from a clustering that found nothing. Finding 41.
    active_latent_units: int = 0
    #: How far the latent means sit from the origin, root mean square. Falling
    #: towards zero alongside active_latent_units is the signature of collapse.
    latent_rms: float = 0.0
    #: How many latent dimensions there are, so the count above has a scale.
    latent_dim: int = 0

    #: The largest total gradient norm over the epoch's steps, measured before
    #: clipping, and the mean. Zero on a measuring pass. Issue #22's excursion
    #: had only ever been seen as a loss sampled every 800 epochs; these say
    #: whether the gradients blew up, which points at a different cause than a
    #: loss that jumped on ordinary gradients.
    largest_gradient_norm: float = 0.0
    mean_gradient_norm: float = 0.0
    #: The largest absolute change to any parameter in the epoch's last step:
    #: what the optimiser did, as opposed to what the gradient asked for. Adam
    #: can turn a small gradient into a large step once its second-moment
    #: estimate has decayed over a long flat stretch. One snapshot per epoch,
    #: so it costs one copy of the parameters rather than one per step.
    largest_update: float = 0.0
    #: Mean relative error on the entries hidden from the model, measured on
    #: the training split after the epoch when early stopping watches it
    #: (``TrainConfig.stop_on``), None otherwise. The number ``exp/10`` and
    #: ``exp/11`` score.
    imputation_error: float | None = None

    def __post_init__(self) -> None:
        if self.n_batches < 1:
            raise ConfigError("a split with no batches has no mean to report")

    @property
    def latent_collapsed(self) -> bool:
        """True when no latent dimension varies between samples."""
        return self.latent_dim > 0 and self.active_latent_units == 0


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One epoch, with the batch counts the means were taken over."""

    epoch: int
    train: SplitResult
    validation: SplitResult | None = None

    @property
    def watched_loss(self) -> float:
        """The loss early stopping looks at: validation if there is one."""
        return self.train.mean_loss if self.validation is None else self.validation.mean_loss

    def watched_value(self, stop_on: str) -> float:
        """The figure early stopping looks at, for the rule in force."""
        if stop_on == "imputation":
            if self.train.imputation_error is None:
                raise RuntimeError(
                    "stop_on='imputation' but no imputation error was measured this epoch"
                )
            return self.train.imputation_error
        return self.watched_loss


@dataclass
class TrainingHistory:
    """Every epoch, and why training finished."""

    epochs: list[EpochRecord] = field(default_factory=list)
    stopped_early: bool = False
    best_epoch: int | None = None
    #: True when early stopping had to watch the training loss for want of a
    #: validation split. Recorded because it weakens what early stopping means.
    watched_training_loss: bool = False
    #: What early stopping watched: "loss" or "imputation" (``TrainConfig.stop_on``).
    watched: str = "loss"
    #: True if the latent space ever collapsed during this run, and when.
    latent_collapsed: bool = False
    first_collapsed_epoch: int | None = None

    def __len__(self) -> int:
        return len(self.epochs)

    @property
    def train_losses(self) -> list[float]:
        return [e.train.mean_loss for e in self.epochs]

    @property
    def validation_losses(self) -> list[float]:
        return [
            e.validation.mean_loss for e in self.epochs if e.validation is not None
        ]



#: A latent dimension counts as active when its mean varies between samples by
#: more than this. Below it the dimension is telling you nothing about which
#: sample is which.
ACTIVE_LATENT_VARIANCE = 0.01


def _latent_means(output) -> torch.Tensor:
    """The latent means from one forward pass, as one tensor.

    All three models report one summary per sample: the full model combines
    its tissues into a single summary before decoding, so there is one to
    report however many tissues went in. A list is joined along the feature
    axis rather than refused, so that a model which reports one per tissue
    would be read rather than crash here.
    """
    means = output.latent_mean
    if isinstance(means, (list, tuple)):
        return torch.cat([m.detach() for m in means], dim=1)
    return means.detach()


def _latent_health(seen: list[torch.Tensor]) -> tuple[int, float, int]:
    """How many latent dimensions vary, and how far they sit from the origin.

    Taken over the whole split rather than per batch, because the variance of a
    small batch is noisy and a dimension can look dead in one batch and not in
    the split.
    """
    if not seen:
        return 0, 0.0, 0
    stacked = torch.cat(seen, dim=0)
    if stacked.shape[0] < 2:  # variance needs two samples
        return stacked.shape[1], float(stacked.pow(2).mean().sqrt()), stacked.shape[1]
    active = int((stacked.var(dim=0) > ACTIVE_LATENT_VARIANCE).sum())
    return active, float(stacked.pow(2).mean().sqrt()), stacked.shape[1]


def _gradient_norm(model: nn.Module, clip: float | None) -> float:
    """The total gradient norm before clipping, clipping on the way if asked.

    ``clip_grad_norm_`` returns the norm it found before scaling, so the
    measurement is free when clipping is on. When it is off the norm is
    computed the same way, so both settings record the same quantity.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
    if clip is not None:
        return float(nn.utils.clip_grad_norm_(parameters, clip))
    return float(torch.norm(torch.stack([p.grad.detach().norm() for p in parameters])))


def run_epoch(
    model: nn.Module,
    data: TensorData | MultiTissueData,
    config: PreffectConfig,
    epoch: int,
    optimiser: torch.optim.Optimizer | None = None,
    rng: np.random.Generator | None = None,
) -> SplitResult:
    """One pass over one split.

    Training and validation share this function deliberately. Version 0.1 wrote
    the two passes out separately, and they drifted: the validation one divided
    by the wrong thing.

    :param data: one split for the simple and single models, or several
        matched tissues for the full model. The two are told apart by type
        rather than by asking a batch what it carries: asking is what stopped
        every full-model run at its first batch (finding 55).
    :param optimiser: supply it to learn, omit it to only measure.
    """
    learning = optimiser is not None
    model.train(learning)

    batches = batch_indices(
        data.n_samples,
        config.train.batch_size,
        rng=rng,
        shuffle=learning and rng is not None,
        drop_last=config.train.drop_last_batch if learning else False,
    )
    if not batches:
        raise ConfigError(
            f"a split of {data.n_samples} samples produced no batches at a batch "
            f"size of {config.train.batch_size}"
        )

    running_total = 0.0
    running_terms: dict[str, float] = {}
    gradient_norms: list[float] = []
    largest_update = 0.0

    seen_latents: list[torch.Tensor] = []
    with torch.set_grad_enabled(learning):
        for index, positions in enumerate(batches):
            part = data.select(positions)
            # The library-size correction for this batch's rows, None when
            # nothing is hidden. Without it a masked batch is scaled by a total
            # short by the hidden fraction while its targets are not, which is
            # finding 56.
            if isinstance(part, MultiTissueData):
                # Several matched tissues: the model takes the list of counts,
                # the target tissue's corrections and one network per tissue,
                # and predicts the target tissue alone. The same shape ``infer``
                # uses on the same data. Until 2026-11-02 this branch did not
                # exist and the loop asked every batch ``has_network``, which a
                # multi-tissue batch does not answer, so the full model stopped
                # at its first batch and had never been trained. Finding 55.
                output = model(
                    [t.inputs for t in part.tissues],
                    part.target.corrections,
                    [t.edge_index for t in part.tissues],
                    library_scale=part.target.library_scale,
                )
                # One observed network per tissue. compute_loss takes a list
                # and gives each tissue its own named term and weight.
                network = [t.adjacency_matrix() for t in part.tissues]
                targets = part.target.targets
                # Which of this batch's rows are real samples, when some
                # patients were missing the predicted tissue and had a row
                # padded in. None when none were, so a complete cohort is
                # scored exactly as it was before padding existed.
                real_rows = part.target_real_rows()
            elif part.has_network:
                output = model(
                    part.inputs,
                    part.corrections,
                    part.edge_index,
                    library_scale=part.library_scale,
                )
                network = part.adjacency_matrix()
                targets = part.targets
                real_rows = None
            else:
                output = model(
                    part.inputs, part.corrections, library_scale=part.library_scale
                )
                network = None
                targets = part.targets
                real_rows = None

            loss = compute_loss(
                output,
                targets,
                weights=config.loss,
                delays=config.delays,
                epoch=epoch,
                batch_labels=None,
                network=network,
                real_rows=real_rows,
            )

            if learning:
                optimiser.zero_grad(set_to_none=True)
                loss.total.backward()
                gradient_norms.append(_gradient_norm(model, config.train.gradient_clip_norm))
                last_step = index == len(batches) - 1
                if last_step:
                    tracked = [p for p in model.parameters() if p.grad is not None]
                    before = [p.detach().clone() for p in tracked]
                optimiser.step()
                if last_step:
                    pairs = zip(tracked, before, strict=True)
                    largest_update = max(
                        (float((p.detach() - b).abs().max()) for p, b in pairs), default=0.0
                    )

            seen_latents.append(_latent_means(output))
            running_total += float(loss.total.detach())
            for name, value in loss.named_values().items():
                running_terms[name] = running_terms.get(name, 0.0) + value

    n = len(batches)
    active, rms, latent_dim = _latent_health(seen_latents)
    return SplitResult(
        active_latent_units=active,
        latent_rms=rms,
        latent_dim=latent_dim,
        largest_gradient_norm=max(gradient_norms, default=0.0),
        mean_gradient_norm=sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0,
        largest_update=largest_update,
        mean_loss=running_total / n,
        mean_terms={name: total / n for name, total in running_terms.items()},
        n_batches=n,
        n_samples=sum(len(b) for b in batches),
    )


def hidden_entry_scores(model: nn.Module, data: TensorData) -> dict:
    """Everything ``imputation_error`` reports on the entries hidden from the model.

    ``hidden`` is the mean relative error; ``n_hidden`` how many entries it was
    computed over, and ``zero_truth_entries_skipped`` how many of those were
    left out because their true value is zero. Reported together, because an
    error with no count beside it does not say what it was computed over
    (decision of 2026-10-29 on masking).
    """
    from preffect.inference import imputation_error, infer

    if getattr(data, "mask", None) is None:
        raise ConfigError(
            "nothing is hidden in this data, so there is no imputation error to measure"
        )
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        result = infer(model, data)
    expected = result.expected_counts.detach().cpu()
    hidden = torch.from_numpy(data.mask.hidden)
    return imputation_error(
        expected, data.targets.detach().cpu(), hidden=hidden, kind="mean_relative"
    )


def hidden_entry_error(model: nn.Module, data: TensorData) -> float:
    """Mean relative error on the entries hidden from the model, over a whole split.

    The number ``exp/10`` and ``exp/11`` score: inference in evaluation mode
    with the latent mean, compared with the targets at the hidden entries. The
    forward pass draws random numbers even when the draw is not used, so the
    generator's state is restored afterwards: a run that measures this every
    epoch is the same run as one that does not, stopped sooner or later.
    """
    return hidden_entry_scores(model, data)["hidden"]


def train(
    model: nn.Module,
    train_data: TensorData | MultiTissueData,
    config: PreffectConfig,
    validation_data: TensorData | MultiTissueData | None = None,
    on_epoch=None,
) -> TrainingHistory:
    """Fit a model.

    :param train_data: one split for the simple and single models, or several
        matched tissues for the full model.
    :param on_epoch: called with each :class:`EpochRecord` as it is completed,
        for progress reporting. Reporting is the caller's business; this
        function does not print.
    """
    # The weights were drawn when the model was built, at whatever thread count
    # was in force then. Setting the count now changes nothing about them, but
    # a model built under one count and trained under another is not the model
    # a second call in the same process would build: initialisation sums floats
    # in a different order at a different count, and on this objective that
    # last-bit difference can grow into a different model (issues #22, #23).
    # Nothing can be re-drawn here, so say so, once, naming both counts.
    wanted = config.runtime.num_threads
    if wanted is not None and wanted != torch.get_num_threads():
        warnings.warn(
            f"the model was built while PyTorch was using {torch.get_num_threads()} "
            f"thread(s) and this run will use {wanted}. Starting weights drawn at "
            f"different thread counts differ in their last bits, and on this "
            f"objective that can grow into a different model, so this run may not "
            f"match one where configure_runtime(config) was called before the model "
            f"was built. Call it first. Issue #23.",
            RuntimeWarning,
            stacklevel=2,
        )

    configure_runtime(config)
    rng = np.random.default_rng(config.runtime.seed)

    # Everything moves to the requested device here, in one place, so that a
    # run which asked for a GPU runs on one. Until finding 45 nothing did, and
    # ``device="cuda"`` trained on the CPU without saying so.
    device = resolve_device(config.runtime)
    model.to(device)
    train_data = train_data.to(device)
    if validation_data is not None:
        validation_data = validation_data.to(device)

    # Only the parameters that are actually being trained. Normally that is all
    # of them; after ``preffect.pretraining.load_pretrained`` froze part of the
    # model it is not. Handing frozen parameters to Adam is harmless but wastes
    # memory on their moment estimates and makes the optimiser's record of what
    # it is training disagree with the model's.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ConfigError(
            "every parameter in this model is frozen, so there is nothing to "
            "train. Freeze fewer parts, or load the weights without freezing."
        )
    optimiser = torch.optim.Adam(
        trainable,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    history = TrainingHistory(watched=config.train.stop_on)
    watching_loss = config.train.stop_on == "loss" and config.train.uses_early_stopping
    if watching_loss and validation_data is None:
        history.watched_training_loss = True
    if config.train.stop_on == "imputation" and isinstance(train_data, MultiTissueData):
        # Named rather than left to fall through the message below, which
        # would tell someone to hide entries in data where hiding them is not
        # built. ``workflow.hide_entries`` refuses the same thing in the same
        # terms.
        raise ConfigError(
            "stop_on='imputation' watches the error on entries hidden from the "
            "model, and this is the full model's data: one table per tissue, with "
            "hiding entries across them not built. Stop on the loss, or fit one "
            "tissue with the single model and hide entries there."
        )
    if config.train.stop_on == "imputation" and getattr(train_data, "mask", None) is None:
        raise ConfigError(
            "stop_on='imputation' watches the imputation error on the entries hidden "
            "from the model, and this training data hides none. Hide some "
            "(TensorData(..., mask=mcar_mask(...)) in Python, --hide on the command "
            "line), or stop on the loss."
        )

    best_loss = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.train.epochs + 1):
        train_result = run_epoch(model, train_data, config, epoch, optimiser, rng)
        validation_result = (
            None
            if validation_data is None
            else run_epoch(model, validation_data, config, epoch)
        )
        if config.train.stop_on == "imputation":
            train_result = replace(
                train_result, imputation_error=hidden_entry_error(model, train_data)
            )

        record = EpochRecord(epoch=epoch, train=train_result, validation=validation_result)
        history.epochs.append(record)

        # Say it while it is happening. A collapsed latent carries nothing
        # about which sample is which, so everything downstream of it is
        # describing noise, and the loss does not show it: the expression
        # terms carry on falling. Found by measuring after the fact, which is
        # the hard way. Finding 41.
        if record.train.latent_collapsed and not history.latent_collapsed:
            history.latent_collapsed = True
            history.first_collapsed_epoch = record.epoch
            warnings.warn(
                f"the latent space collapsed at epoch {record.epoch}: none of "
                f"its {record.train.latent_dim} dimensions vary between "
                f"samples any more, so it carries no information about which "
                f"sample is which. The loss will keep falling regardless. "
                f"Consider a KL warm-up through LossDelays, or a lower "
                f"learning rate. See issue #20.",
                stacklevel=2,
            )
        if on_epoch is not None:
            on_epoch(record)

        watched = record.watched_value(config.train.stop_on)
        if watched < best_loss - config.train.early_stopping_min_delta:
            best_loss = watched
            history.best_epoch = epoch
            epochs_without_improvement = 0
            if config.train.uses_early_stopping:
                best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1

        if (
            config.train.uses_early_stopping
            and epochs_without_improvement >= config.train.early_stopping_patience
        ):
            history.stopped_early = True
            if best_state is not None:
                model.load_state_dict(best_state)
            break

    return history
