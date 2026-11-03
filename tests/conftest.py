"""Shared helpers for building small count datasets in a temporary folder.

Tests build their own data rather than reading anything from the lab's data
root, so the suite runs anywhere, including on a continuous integration machine
that has no access to it.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def make_counts(
    n_samples: int = 6,
    n_genes: int = 4,
    seed: int = 0,
    sample_names: list[str] | None = None,
) -> ad.AnnData:
    """A small, well-formed count matrix."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 50, size=(n_samples, n_genes)).astype(np.int64)
    # Object dtype, as the package writes them: under pandas 3 a plain list
    # becomes a string array that anndata 0.11 will not write. Finding 48.
    obs = pd.DataFrame(
        {"batch": [i % 2 for i in range(n_samples)]},
        index=pd.Index(sample_names or [f"sample_{i}" for i in range(n_samples)], dtype=object),
    )
    var = pd.DataFrame(index=pd.Index([f"Gene_{j}" for j in range(n_genes)], dtype=object))
    return ad.AnnData(X=counts, obs=obs, var=var)


def write_split(folder: Path, tissues: list[ad.AnnData], stem: str = "counts") -> Path:
    """Write one file per tissue into ``folder``, named the way the loader expects."""
    folder.mkdir(parents=True, exist_ok=True)
    for index, adata in enumerate(tissues, start=1):
        adata.write_h5ad(folder / f"{stem}.tau_{index}.h5ad")
    return folder


@pytest.fixture
def counts():
    return make_counts


@pytest.fixture
def split(tmp_path):
    """Write a split and return its folder."""

    def _write(name: str, tissues: list[ad.AnnData], stem: str = "counts") -> Path:
        return write_split(tmp_path / name, tissues, stem)

    return _write


def pytest_configure(config):
    """Cap PyTorch's threads for the whole suite.

    PyTorch defaults to one thread per core. On the machine this was written on
    that is 112, and for tensors the size the tests use, coordinating them costs
    more than the arithmetic. The graph model tests take 9.1 seconds at 112
    threads and 6.8 at 2.

    ``PREFFECT_TEST_THREADS`` overrides the cap, which is how that comparison is
    made. It exists because an earlier version of this docstring claimed the
    saving was a factor of nine hundred, and there was no way to check it
    without editing this file. See finding 24.
    """
    import os

    import torch

    torch.set_num_threads(int(os.environ.get("PREFFECT_TEST_THREADS", "2")))


def pytest_collection_modifyitems(config, items):
    """Skip ``gpu``-marked tests on a machine with no CUDA device.

    They are not deselected by default, because a GPU test nobody runs is not
    a test; they skip, visibly, where they cannot run. ``PREFFECT_TEST_DEVICE``
    names the device they use, default ``cuda``, which is card 0. On a shared
    machine card 0 is rarely the idle one, and the tests are tiny, so pick the
    quiet card by hand: ``PREFFECT_TEST_DEVICE=cuda:7``.
    """
    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def gpu_device() -> str:
    """The CUDA device a ``gpu`` test should use. See ``PREFFECT_TEST_DEVICE``."""
    import os

    return os.environ.get("PREFFECT_TEST_DEVICE", "cuda")
