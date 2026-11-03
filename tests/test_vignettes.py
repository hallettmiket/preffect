"""The vignette data.

Finding 11: version 0.1's vignettes ship a ``test/`` folder byte-identical to
``validation/``, in all three of simple, single and full, and a published
figure shows a user pointing inference at that folder while the caption calls
it a held-out test set.

These tests exist so that cannot happen again quietly. The generator checks it
too, but a check inside the thing being checked is worth less than one outside.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pytest

pytestmark = pytest.mark.extras  # writes real files, slower than the unit tests

GENERATOR = Path(__file__).resolve().parents[1] / "vignettes" / "make_data.py"
SPLITS = ("train", "validation", "test")


@pytest.fixture(scope="module")
def vignette(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("vignette")
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return out


def read(folder: Path) -> ad.AnnData:
    return ad.read_h5ad(next(folder.glob("*.h5ad")))


@pytest.mark.parametrize("kind", ["simple", "single"])
def test_no_sample_is_in_two_splits(vignette, kind):
    """The property version 0.1's vignettes lacked."""
    names = {s: set(map(str, read(vignette / kind / s).obs_names)) for s in SPLITS}
    assert names["train"] & names["validation"] == set()
    assert names["train"] & names["test"] == set()
    assert names["validation"] & names["test"] == set()


@pytest.mark.parametrize("kind", ["simple", "single"])
def test_the_three_files_are_not_copies_of_each_other(vignette, kind):
    """Checked by checksum, which is how the defect was found in 0.1."""
    digests = {
        s: hashlib.md5(next((vignette / kind / s).glob("*.h5ad")).read_bytes()).hexdigest()
        for s in SPLITS
    }
    assert len(set(digests.values())) == 3, f"two splits are the same file: {digests}"


@pytest.mark.parametrize("kind", ["simple", "single"])
def test_every_split_carries_the_same_genes(vignette, kind):
    """Otherwise applying a model across them matches on a different column."""
    genes = [tuple(map(str, read(vignette / kind / s).var_names)) for s in SPLITS]
    assert genes[0] == genes[1] == genes[2]


@pytest.mark.parametrize("kind", ["simple", "single"])
def test_counts_are_whole_numbers(vignette, kind):
    """PREFFECT models counts; a normalised matrix would not fit."""
    import numpy as np

    for split in SPLITS:
        values = read(vignette / kind / split).X
        assert np.all(values >= 0)
        assert np.allclose(values, np.round(values))


def test_the_single_vignette_has_a_network_with_real_edges(vignette):
    """Finding 35: an identity matrix is a network with nothing in it."""
    import numpy as np

    network = np.asarray(read(vignette / "single" / "train").obsm["sample_sample_adj"])
    off_diagonal = network.copy()
    np.fill_diagonal(off_diagonal, 0)
    assert np.count_nonzero(off_diagonal) > 0, "no edges between samples"


def test_the_simple_vignette_has_no_network(vignette):
    assert "sample_sample_adj" not in read(vignette / "simple" / "train").obsm


@pytest.mark.parametrize("kind", ["simple", "single"])
def test_there_is_a_group_signal_to_find(vignette, kind):
    """A vignette where the model cannot find anything teaches nothing."""
    adata = read(vignette / kind / "train")
    assert "subtype" in adata.obs
    assert len(set(adata.obs["subtype"])) >= 2


def test_the_generator_refuses_to_write_identical_splits(tmp_path, monkeypatch):
    """The generator's own guarantee, exercised by breaking it.

    Forces every split to the same samples, by stopping the sample-name
    counter advancing, and confirms the generator raises rather than writing
    them, which is what version 0.1's vignettes needed and lacked.
    """
    from preffect import example_data
    from preffect.errors import DataError

    monkeypatch.setattr(example_data, "_advance", lambda offset, n: offset)
    with pytest.raises(DataError, match="not distinct"):
        example_data.write_example_data(tmp_path / "out")


def test_the_script_and_the_command_write_the_same_files(vignette, tmp_path):
    """One generator behind both, so the page and the notebook cannot drift."""
    from preffect.cli import main

    out = tmp_path / "by_command"
    assert main(["example-data", "--out", str(out)]) == 0
    for kind in ("simple", "single", "full"):
        for split in SPLITS:
            a = hashlib.md5((vignette / kind / split / "counts.tau_1.h5ad").read_bytes())
            b = hashlib.md5((out / kind / split / "counts.tau_1.h5ad").read_bytes())
            assert a.hexdigest() == b.hexdigest(), f"{kind}/{split} differs"


NOTEBOOK = Path(__file__).resolve().parents[1] / "vignettes" / "preffect_vignette.ipynb"


def test_the_notebook_runs_the_worked_example(tmp_path, monkeypatch):
    """The notebook mirrors the vignette page, so it is run the way the page is.

    Every code cell is executed in order in an empty folder, without Jupyter:
    the cells call the package, not a kernel. What the page promises at the end
    has to be there: a saved model and a cleaned file with the latent summary.
    """
    import json

    cells = json.loads(NOTEBOOK.read_text())["cells"]
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for index, cell in enumerate(c for c in cells if c["cell_type"] == "code"):
        source = "".join(cell["source"])
        exec(compile(source, f"notebook cell {index}", "exec"), namespace)  # noqa: S102
    assert (tmp_path / "model.pt").is_file()
    cleaned = ad.read_h5ad(tmp_path / "cleaned.h5ad")
    assert cleaned.n_obs == 40
    assert "latent" in cleaned.obsm
