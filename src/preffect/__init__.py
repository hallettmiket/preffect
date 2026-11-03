"""PREFFECT: probabilistic modelling of RNA sequencing from FFPE material.

FFPE stands for formalin-fixed paraffin-embedded, the way tissue samples have
been preserved in pathology archives for a century. RNA extracted from these
blocks is degraded, so sequencing them produces counts with many missing values
and extreme outliers. PREFFECT fits a generative model to those counts while
adjusting for technical and biological variables, so that the cleaned values can
be used with ordinary downstream tools.

This is version 0.2, a rewrite. The method is described in Mucaki et al.,
Journal of Translational Medicine 23:1023 (2025),
https://doi.org/10.1186/s12967-025-07031-y
"""

from __future__ import annotations

from preffect.config import (
    CorrectionVariable,
    DataConfig,
    Dispersion,
    Likelihood,
    LossWeights,
    Masking,
    ModelConfig,
    ModelKind,
    PreffectConfig,
    RuntimeConfig,
    TrainConfig,
)
from preffect.errors import (
    CheckpointError,
    ConfigError,
    DataError,
    FitError,
    PreffectError,
)

__version__ = "0.2.0"

__all__ = [
    "CheckpointError",
    "ConfigError",
    "CorrectionVariable",
    "DataConfig",
    "DataError",
    "Dispersion",
    "FitError",
    "Likelihood",
    "LossWeights",
    "Masking",
    "ModelConfig",
    "ModelKind",
    "PreffectConfig",
    "PreffectError",
    "RuntimeConfig",
    "TrainConfig",
    "__version__",
]
