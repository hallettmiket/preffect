"""The steps the command line and the interface share.

Each is also exercised through the command line's own tests, since the
command line now calls these. What is tested here is the part the interface
needs and the command line does not: a report as a value, and an apply that
is silent unless asked to speak.
"""

from __future__ import annotations

import pytest
from tests.conftest import make_counts, write_split

from preffect.cli import main
from preffect.workflow import apply_checkpoint, inspect_split


@pytest.fixture
def folder(tmp_path):
    return write_split(tmp_path / "train", [make_counts(n_samples=24, n_genes=12, seed=0)])


def test_inspect_split_reports_a_usable_folder(folder):
    report = inspect_split(folder)
    assert report.usable
    assert [(t.samples, t.genes) for t in report.tissues] == [(24, 12)]
    assert report.tissues[0].network == "no network"


def test_inspect_split_reports_a_missing_folder_as_a_problem_not_an_exception(tmp_path):
    report = inspect_split(tmp_path / "absent")
    assert not report.usable
    assert report.tissues == ()
    assert "no such folder" in report.problem


def test_inspect_split_reports_counts_that_would_stop_a_run(tmp_path):
    adata = make_counts(n_samples=6, n_genes=4)
    adata.var_names = ["g", "g", "h", "i"]  # a repeated gene name, finding 34
    report = inspect_split(write_split(tmp_path / "dup", [adata]))
    assert not report.usable
    assert report.tissues, "it should still say what was there"
    assert "appear more than once" in report.problem


def test_apply_checkpoint_runs_a_saved_model_and_matches_genes(folder, tmp_path):
    model = tmp_path / "m.pt"
    assert main(["train", "--train", str(folder), "--out", str(model),
                 "--epochs", "2", "--batch-size", "8", "--seed", "0", "--threads", "2"]) == 0
    lines = []
    result, tissues, checkpoint = apply_checkpoint(model, folder, report=lines.append)
    assert tuple(result.expected_counts.shape) == (24, 12)
    assert tissues[0].n_vars == 12
    assert checkpoint.n_genes == 12
    assert any("matched 12 genes" in line for line in lines)


def test_apply_checkpoint_is_silent_unless_asked(folder, tmp_path, capsys):
    model = tmp_path / "m.pt"
    main(["train", "--train", str(folder), "--out", str(model),
          "--epochs", "2", "--batch-size", "8", "--threads", "2"])
    capsys.readouterr()
    apply_checkpoint(model, folder)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# Hiding entries derives the correction from the mask it just applied.
#
# Finding 56: a library-size correction given by hand wins over one derived
# from a mask, which is what lets a mini-batch carry one. Hiding entries twice
# would otherwise keep the first hiding's correction and scale the second run
# by the wrong amount, silently.
# --------------------------------------------------------------------------


def test_hiding_entries_derives_the_correction_from_the_mask_it_applied():
    import numpy as np
    import torch

    from preffect.data.tensors import TensorData
    from preffect.workflow import hide_entries

    counts = torch.from_numpy(
        np.random.default_rng(0).integers(1, 50, size=(20, 10)).astype(np.float32)
    )
    once = hide_entries(TensorData(inputs=counts.clone(), targets=counts), 0.2, seed=0)
    assert once.library_scale is not None
    np.testing.assert_allclose(
        once.library_scale.numpy(), once.mask.library_scale(), rtol=1e-6
    )

    # Hidden again, and more deeply: the correction has to follow.
    twice = hide_entries(once, 0.5, seed=1)
    np.testing.assert_allclose(
        twice.library_scale.numpy(), twice.mask.library_scale(), rtol=1e-6
    )
    assert float(twice.library_scale.median()) > float(once.library_scale.median())
