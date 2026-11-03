"""Exceptions raised by PREFFECT.

Every one carries a message that names the setting at fault and says what to do
about it. The previous version validated with ``assert``, which Python removes
when run with ``-O``, and which produces a message only a developer can read.
"""

from __future__ import annotations


class PreffectError(Exception):
    """Base class for every error PREFFECT raises deliberately."""


class ConfigError(PreffectError):
    """A configuration is internally inconsistent, or impossible to run.

    Raised while a configuration is being built, before any data is read and
    before a model is constructed, so that a run stops at the point where the
    problem can still be understood.
    """


class DataError(PreffectError):
    """Input data is missing, malformed, or does not match the configuration."""


class CheckpointError(PreffectError):
    """A saved model could not be read, or was written by an incompatible version."""


class FitError(PreffectError):
    """A distribution could not be fitted to the counts it was given.

    The counts were usable but the fit was not: every count was zero, so no
    mean is identified, or no search converged. Version 0.1's standalone
    estimator returned ``None`` in both cases, which a caller fitting a
    distribution per gene reads as an answer only if it thought to check.
    """
