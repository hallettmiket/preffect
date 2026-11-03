"""PREFFECT configuration: typed, validated once, then frozen.

The previous version kept configuration in a single dictionary of about ninety
untyped keys, imported as a module global and modified in place by the data
loader, the batch iterator, the factory and the validation routine. Several
defects came directly from that arrangement, and each rule below exists because
of one of them. The rules are named in ``how_this_project_breaks.md``.

Three properties this module guarantees:

1. **Nothing is silently rewritten.** If a setting cannot be honoured, a
   :class:`~preffect.errors.ConfigError` says which setting, what was asked
   for, and what to do. The previous version would quietly change the model
   you asked for into a different one.
2. **Per-tissue weights are checked against the number of tissues.** A list of
   the wrong length used to be accepted and then silently truncated, which
   dropped whole tissues out of the objective.
3. **Nothing has a filesystem default.** Paths are supplied by the caller or
   the configuration does not build.

A configuration is immutable once constructed. To vary one setting, use
:meth:`PreffectConfig.replace`, which re-runs every check.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from preffect.errors import ConfigError

__all__ = [
    "ModelKind",
    "Likelihood",
    "Dispersion",
    "Masking",
    "CorrectionVariable",
    "DataConfig",
    "ModelConfig",
    "LossWeights",
    "LossDelays",
    "TrainConfig",
    "RuntimeConfig",
    "PreffectConfig",
]


class ModelKind(StrEnum):
    """Which of the three models to build.

    ``SIMPLE`` uses counts alone. ``SINGLE`` adds a sample-sample adjacency
    network for one tissue. ``FULL`` extends that to several matched tissues.
    """

    SIMPLE = "simple"
    SINGLE = "single"
    FULL = "full"

    @property
    def uses_adjacency(self) -> bool:
        return self is not ModelKind.SIMPLE


class Likelihood(StrEnum):
    NB = "NB"
    ZINB = "ZINB"


class Dispersion(StrEnum):
    GENE = "gene"
    GENE_SAMPLE = "gene-sample"
    GENE_BATCH = "gene-batch"


class Masking(StrEnum):
    NONE = "none"
    MCAR = "MCAR"


@dataclass(frozen=True, slots=True)
class CorrectionVariable:
    """A sample annotation the model conditions on, such as sequencing batch."""

    name: str
    kind: str  # "categorical" or "continuous"

    def __post_init__(self) -> None:
        if self.kind not in ("categorical", "continuous"):
            raise ConfigError(
                f"correction variable {self.name!r} has kind {self.kind!r}; "
                "it must be 'categorical' or 'continuous'"
            )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the counts are and how much of them to use.

    ``validation_path`` is optional, and that is deliberate. The previous
    version required a validation folder to exist before it would run, so a
    cohort too small to split was handled by putting the same file in both
    folders. The resulting "validation" loss was the training loss. Here,
    leaving it unset means no validation loss is computed and none is reported.
    """

    train_path: Path
    n_tissues: int = 1
    validation_path: Path | None = None
    test_path: Path | None = None
    select_genes: int | None = None
    select_samples: int | None = None
    masking: Masking = Masking.NONE
    mask_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_path", Path(self.train_path))
        for name in ("validation_path", "test_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))

        if not isinstance(self.masking, Masking):
            try:
                object.__setattr__(self, "masking", Masking(self.masking))
            except ValueError as exc:
                allowed = sorted(m.value for m in Masking)
                raise ConfigError(
                    f"masking must be one of {allowed}, got {self.masking!r}"
                ) from exc

        if self.n_tissues < 1:
            raise ConfigError(f"n_tissues must be at least 1, got {self.n_tissues}")

        for name in ("select_genes", "select_samples"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ConfigError(f"{name} must be at least 1 when set, got {value}")

        if not 0.0 <= self.mask_fraction <= 1.0:
            raise ConfigError(
                f"mask_fraction is a proportion and must lie in [0, 1], got {self.mask_fraction}"
            )
        if self.masking is Masking.NONE and self.mask_fraction > 0:
            raise ConfigError(
                f"mask_fraction is {self.mask_fraction} but masking is 'none'. "
                "Set masking to 'MCAR' to mask counts, or set mask_fraction to 0."
            )
        if self.masking is not Masking.NONE and self.mask_fraction == 0:
            raise ConfigError(
                f"masking is {self.masking.value!r} but mask_fraction is 0, so nothing "
                "would be masked. Set a fraction above 0, or set masking to 'none'."
            )

    @property
    def has_validation(self) -> bool:
        return self.validation_path is not None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """The network itself."""

    kind: ModelKind = ModelKind.SIMPLE
    likelihood: Likelihood = Likelihood.NB
    dispersion: Dispersion = Dispersion.GENE_SAMPLE
    latent_dim: int = 20
    hidden_dim: int = 50
    embed_dim: int = 5
    attention_heads: int = 8
    leaky_relu_slope: float = 0.2
    dropout: float = 0.1
    infer_library_size: bool = False
    correction_variables: tuple[CorrectionVariable, ...] = ()

    # The dispersion is produced on a log scale and exponentiated, which keeps
    # it positive without a softplus. Version 0.1 called this theta_transform
    # and reported that it helps when the dispersion is above one.
    log_scale_dispersion: bool = True

    # Ceilings that stop a single extreme value turning into an infinity part
    # way through a pass. The values are version 0.1's. They are settings
    # rather than constants because the right ceiling depends on the depth of
    # the data, and a run that hits one is telling you something.
    max_latent_mean: float = 162754.0  # exp(12)
    max_log_variance: float = 10.0
    max_log_library_size: float = 20.0

    def __post_init__(self) -> None:
        # Accept the string form of each enum, so that settings read back from
        # a checkpoint, where they are stored as plain strings, rebuild into
        # the same object. Also makes ModelConfig(kind="simple") work.
        for name, enum_type in (
            ("kind", ModelKind),
            ("likelihood", Likelihood),
            ("dispersion", Dispersion),
        ):
            value = getattr(self, name)
            if not isinstance(value, enum_type):
                try:
                    object.__setattr__(self, name, enum_type(value))
                except ValueError as exc:
                    allowed = sorted(m.value for m in enum_type)
                    raise ConfigError(
                        f"{name} must be one of {allowed}, got {value!r}"
                    ) from exc

        if self.hidden_dim <= self.latent_dim:
            raise ConfigError(
                f"hidden_dim ({self.hidden_dim}) must be larger than latent_dim "
                f"({self.latent_dim}); the hidden layer is the wider one"
            )
        for name in ("latent_dim", "hidden_dim", "embed_dim", "attention_heads"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout is a proportion in [0, 1), got {self.dropout}")
        for name in ("max_latent_mean", "max_log_variance", "max_log_library_size"):
            if getattr(self, name) <= 0:
                raise ConfigError(
                    f"{name} is a ceiling on a magnitude and must be positive, "
                    f"got {getattr(self, name)}"
                )

        names = [v.name for v in self.correction_variables]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ConfigError(f"correction variables named more than once: {sorted(duplicates)}")

        if self.dispersion is Dispersion.GENE_BATCH and "batch" not in names:
            raise ConfigError(
                "dispersion is 'gene-batch', which estimates one dispersion per gene per "
                "batch, but 'batch' is not among the correction variables "
                f"{names or '(none)'}. Add it, or choose another dispersion setting."
            )

    @property
    def corrects_variables(self) -> bool:
        return len(self.correction_variables) > 0


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Relative weight of each term in the objective.

    Terms that apply once per tissue are given one weight per tissue. The
    previous version built these lists at import time from a tissue count of 1,
    nothing rebuilt them when the count changed, and the weighting step paired
    weights against losses with ``zip``, which stops at the shorter of the two.
    A two-tissue run with a one-entry list therefore dropped the second tissue
    from the objective entirely, with no error. Here the lengths are checked.
    """

    expression_recon: float = 100.0
    library_recon: tuple[float, ...] = (100.0,)
    adjacency_recon: tuple[float, ...] = (100.0,)
    expression_kl: tuple[float, ...] = (0.1,)
    library_kl: tuple[float, ...] = (0.1,)
    adjacency_kl: tuple[float, ...] = (0.1,)
    batch_centroid: tuple[float, ...] = (1.0,)

    # The objective is divided by the sum of the weights of the terms that are
    # actually active, which makes it a weighted mean. The previous version
    # summed the weights of every term named in the configuration, including
    # terms the chosen model does not have and terms whose value was
    # structurally zero, so the reported loss was not the mean it appeared to
    # be. Set this to False to report the unnormalised sum.
    normalise_by_weight_sum: bool = True

    #: How the network reconstruction term treats the pairs that are not
    #: linked. These networks are 0.2 to 8 per cent dense, so the absent pairs
    #: are most of the term. ``"balanced"``, the default, weights each
    #: linked pair up by the number of absent pairs per linked pair, so the two
    #: kinds carry half the term each. ``"none"`` scores every entry of the
    #: batch's network equally, as version 0.1 did and as every graph-model
    #: result before 2026-10-30 was produced; under it the network half learns
    #: nothing, measured in ``exp/17``: the decoder settles on predicting no
    #: edge anywhere and the network latent collapses onto its prior in every
    #: seed. A third way, scoring the linked pairs against as many absent pairs
    #: drawn at random, lost that comparison and lives in ``exp/17`` rather
    #: than here. On the two datasets run since, the DCIS cohort in ``exp/17``
    #: and ``dataset_full_1`` in ``exp/18``, balancing did not recover a
    #: network; see the module docstring of :mod:`preffect.loss`. Decision of
    #: 2026-10-30 on weighting the network term.
    edge_balance: str = "balanced"

    #: Names of the weights that take one value per tissue. A class constant,
    #: not a setting, so it is not a constructor argument.
    PER_TISSUE: ClassVar[tuple[str, ...]] = (
        "library_recon",
        "adjacency_recon",
        "expression_kl",
        "library_kl",
        "adjacency_kl",
        "batch_centroid",
    )

    #: The values ``edge_balance`` accepts.
    EDGE_BALANCE_MODES: ClassVar[tuple[str, ...]] = ("none", "balanced")

    def __post_init__(self) -> None:
        for name in self.PER_TISSUE:
            value = getattr(self, name)
            if isinstance(value, (int, float)):
                value = (float(value),)
            object.__setattr__(self, name, tuple(float(v) for v in value))

        if self.expression_recon <= 0:
            raise ConfigError(
                f"expression_recon must be a positive number, got {self.expression_recon}"
            )
        if self.edge_balance not in self.EDGE_BALANCE_MODES:
            raise ConfigError(
                f"edge_balance must be one of {list(self.EDGE_BALANCE_MODES)}, got "
                f"{self.edge_balance!r}. 'balanced' gives the linked pairs the same "
                f"total weight as the absent ones; 'none' scores every pair of the "
                f"network equally, as version 0.1 did."
            )
        for name in self.PER_TISSUE:
            for i, v in enumerate(getattr(self, name)):
                if v <= 0:
                    raise ConfigError(
                        f"{name}[{i}] must be a positive number, got {v}"
                    )

    def check_tissue_count(self, n_tissues: int) -> None:
        """Every per-tissue list must have exactly one entry per tissue."""
        wrong = {
            name: len(getattr(self, name))
            for name in self.PER_TISSUE
            if len(getattr(self, name)) != n_tissues
        }
        if wrong:
            detail = ", ".join(f"{k} has {v}" for k, v in sorted(wrong.items()))
            raise ConfigError(
                f"these loss weights need one entry per tissue, and there are "
                f"{n_tissues} tissues, but {detail}. Give each one "
                f"{n_tissues} value{'s' if n_tissues != 1 else ''}."
            )

    def for_tissue_count(self, n_tissues: int) -> LossWeights:
        """Return a copy with every per-tissue weight repeated to the right length.

        A convenience for the common case of wanting the same weight on every
        tissue. It refuses lists that are neither length 1 nor already correct,
        rather than guessing what was meant.
        """
        updates: dict[str, tuple[float, ...]] = {}
        for name in self.PER_TISSUE:
            value = getattr(self, name)
            if len(value) == n_tissues:
                continue
            if len(value) == 1:
                updates[name] = value * n_tissues
            else:
                raise ConfigError(
                    f"{name} has {len(value)} entries and cannot be expanded to "
                    f"{n_tissues} tissues; give either 1 value or {n_tissues}"
                )
        return dataclasses.replace(self, **updates) if updates else self


@dataclass(frozen=True, slots=True)
class LossDelays:
    """The first epoch at which each term joins the objective.

    Introducing a term late lets the model settle on an easier problem before a
    harder one is added. The supplementary calls these "loss delays". A value
    of 1 means the term is active from the start, which is what every published
    run used.

    Epochs are counted from 1, so a delay of 3 means the term is absent for
    epochs 1 and 2 and present from epoch 3.
    """

    expression_recon: int = 1
    expression_kl: int = 1
    library_recon: int = 1
    library_kl: int = 1
    adjacency_recon: int = 1
    adjacency_kl: int = 1
    batch_centroid: int = 1

    def __post_init__(self) -> None:
        for field_ in dataclasses.fields(self):
            value = getattr(self, field_.name)
            if value < 1:
                raise ConfigError(
                    f"{field_.name} starts at epoch {value}; epochs count from 1, "
                    f"so use 1 for a term that is active from the start"
                )

    def active_at(self, term: str, epoch: int) -> bool:
        """Is this term part of the objective at this epoch?

        A per-tissue term carries its tissue number as a suffix, for example
        ``adjacency_recon_2``. The delay is set once for the term and applies
        to every tissue, so the suffix is stripped before looking it up.
        """
        base = term.rsplit("_", 1)
        if len(base) == 2 and base[1].isdigit() and hasattr(self, base[0]):
            term = base[0]
        if not hasattr(self, term):
            raise ConfigError(
                f"no delay is defined for a loss term called {term!r}. "
                f"Known terms: {sorted(f.name for f in dataclasses.fields(self))}"
            )
        return epoch >= getattr(self, term)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """How the model is fitted."""

    epochs: int = 1000
    batch_size: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    gradient_clip_norm: float | None = 10.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    #: What early stopping watches. ``"loss"``: the validation loss, or the
    #: training loss without a validation split. ``"imputation"``: the mean
    #: relative error on the entries hidden from the model, measured on the
    #: training split after every epoch. The loss on unseen samples bottoms out
    #: about ten times sooner than imputation of the fitted cohort settles
    #: (finding 50), so the choice is which use the model is for: new samples,
    #: or the cohort it was fitted on. Decision of 2026-10-28 on training
    #: length, and issue #28.
    stop_on: str = "loss"

    # The previous version silently discarded the final incomplete batch of
    # every epoch, so with 250 samples and a batch of 200, fifty samples never
    # entered training and nothing said so. Here the default keeps every
    # sample, and dropping them is something you ask for.
    drop_last_batch: bool = False

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ConfigError(f"weight_decay cannot be negative, got {self.weight_decay}")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ConfigError(
                "gradient_clip_norm must be positive when set, or None to disable "
                f"clipping; got {self.gradient_clip_norm}"
            )
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ConfigError(
                "early_stopping_patience must be at least 1 when set, or None to "
                f"disable early stopping; got {self.early_stopping_patience}"
            )

        if self.stop_on not in ("loss", "imputation"):
            raise ConfigError(f"stop_on must be 'loss' or 'imputation', got {self.stop_on!r}")
        if self.stop_on == "imputation" and self.early_stopping_patience is None:
            raise ConfigError(
                "stop_on='imputation' only means something with early_stopping_patience; "
                "without it no run stops early and the setting would be ignored"
            )

    @property
    def uses_early_stopping(self) -> bool:
        return self.early_stopping_patience is not None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Where the work runs and whether it can be repeated.

    ``device`` is checked here and acted on in one place,
    :func:`preffect.training.resolve_device`, which ``train`` and ``apply``
    both call before reading any data: it moves the model and the data there,
    and refuses by name a device the machine does not have. The previous
    version rebuilt the device from the configuration in six separate places
    and two of those did not honour the request for a CPU, so asking for a CPU
    run on a machine with a GPU moved some tensors to the GPU anyway and the
    run failed on a device mismatch. The first version of this class went the
    other way: it validated the string and nothing ever moved anything, so a
    run that asked for a GPU trained on the CPU without a word. Finding 45.
    """

    device: str = "cpu"
    seed: int | None = None
    deterministic: bool = False
    num_threads: int | None = None

    #: 'cpu', 'cuda', or 'cuda:' followed by an index. Anything else is a typo,
    #: and 'cuda0' is the one worth naming: it looks right and is not.
    _DEVICE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(cpu|cuda(:\d+)?)$")

    def __post_init__(self) -> None:
        device = str(self.device)
        if not self._DEVICE_PATTERN.match(device):
            raise ConfigError(
                f"device must be 'cpu' or 'cuda', optionally with an index such as "
                f"'cuda:0'; got {device!r}"
            )
        object.__setattr__(self, "device", device)

        if self.deterministic and self.seed is None:
            raise ConfigError(
                "deterministic is True but seed is None, so runs would still differ. "
                "Set a seed, or set deterministic to False."
            )
        if self.num_threads is not None and self.num_threads < 1:
            raise ConfigError(f"num_threads must be at least 1 when set, got {self.num_threads}")

    @property
    def uses_cuda(self) -> bool:
        return self.device.startswith("cuda")


@dataclass(frozen=True, slots=True)
class PreffectConfig:
    """A complete, checked configuration for one run."""

    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    delays: LossDelays = field(default_factory=LossDelays)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        n = self.data.n_tissues
        kind = self.model.kind

        # The model you ask for is the model you get. The previous version
        # rewrote the request instead: asking for the single-tissue graph model
        # produced the plain model, because the check ran before the data was
        # read and so could not yet see that an adjacency network was present.
        if kind is ModelKind.FULL and n < 2:
            raise ConfigError(
                "the 'full' model combines several matched tissues, so n_tissues must "
                f"be at least 2; got {n}. Use 'single' for one tissue with an "
                "adjacency network, or 'simple' for one tissue without one."
            )
        if kind is not ModelKind.FULL and n != 1:
            raise ConfigError(
                f"the {kind.value!r} model handles exactly one tissue, but n_tissues "
                f"is {n}. Use 'full' for several tissues."
            )

        self.loss.check_tissue_count(n)

    @property
    def n_tissues(self) -> int:
        return self.data.n_tissues

    def replace(self, **changes: object) -> PreffectConfig:
        """Return a new configuration with sections replaced, fully re-checked."""
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ConfigError(
                f"unknown configuration section(s): {sorted(unknown)}. "
                f"Expected any of: {sorted(f.name for f in dataclasses.fields(self))}"
            )
        return dataclasses.replace(self, **changes)
