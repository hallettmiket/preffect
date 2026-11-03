"""The command line.

Every command is run end to end and checked for output. That is the whole
remedy for finding 11: version 0.1's command line read its arguments, built a
session object and exited without running anything, so a run appeared to
succeed and produced no file. No amount of care in the code prevents that; a
test that asserts a file exists afterwards does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tests.conftest import make_counts, write_split

from preffect.cli import build_parser, main


@pytest.fixture
def counts_folder(tmp_path):
    """A small, well-formed training set on disk."""

    def _make(name: str, seed: int = 0, n_samples: int = 24, with_network: bool = False):
        adata = make_counts(n_samples=n_samples, n_genes=12, seed=seed)
        if with_network:
            n = adata.n_obs
            network = np.zeros((n, n), dtype=np.int64)
            for i in range(n):  # a ring, so nothing is isolated
                network[i, (i + 1) % n] = 1
                network[(i + 1) % n, i] = 1
            adata.obsm["sample_sample_adj"] = network
        return write_split(tmp_path / name, [adata])

    return _make


# --------------------------------------------------------------------------
# Each command runs and produces what it promises.
# --------------------------------------------------------------------------


def test_check_reports_what_it_found(counts_folder, capsys):
    assert main(["check", "--train", str(counts_folder("train"))]) == 0
    printed = capsys.readouterr().out
    assert "24 samples, 12 genes" in printed
    assert "the counts look usable" in printed


def test_check_says_when_there_is_no_validation_set(counts_folder, capsys):
    main(["check", "--train", str(counts_folder("train"))])
    assert "no validation loss will be reported" in capsys.readouterr().out


def matched_split(folder: Path, missing: tuple[int, ...] = ()) -> Path:
    """Two tissues of six patients, with the named patients absent from the second.

    The shape a cohort takes when some patients have no matched normal: the
    correspondence table has an empty cell for them and the second file has no
    row for them at all.
    """
    import pandas as pd

    names = [[f"tau{t}_{i}" for i in range(6)] for t in (1, 2)]
    table = pd.DataFrame({"tau_1": names[0], "tau_2": names[1]})
    # The empty cell is the string "NA", which the reader treats as missing;
    # a float NaN in a column of names is an object column under pandas 2 that
    # anndata refuses to write, which is how this test failed on Python 3.11.
    for patient in missing:
        table.loc[patient, "tau_2"] = "NA"

    tissues = []
    for t, own in enumerate(names):
        rows = [i for i in range(6) if not (t == 1 and i in missing)]
        kept = [own[i] for i in rows]
        adata = make_counts(n_samples=len(kept), n_genes=12, seed=t)
        adata.obs_names = pd.Index(kept, dtype=object)
        frame = table.iloc[rows].copy()
        frame.index = pd.Index(kept, dtype=object)
        adata.obsm["sample_association"] = frame
        tissues.append(adata)
    return write_split(folder, tissues)


def test_check_reports_how_many_patients_have_no_sample_in_each_tissue(tmp_path, capsys):
    """Issue #14: the full model pads those patients in, so the count is worth seeing."""
    folder = matched_split(tmp_path / "gapped", missing=(1, 4))
    assert main(["check", "--train", str(folder), "--tissues", "2"]) == 0
    printed = capsys.readouterr().out
    assert "tissue 1: every patient has a sample here" in printed
    assert "tissue 2: 2 patient(s) have no sample here" in printed
    assert "padded in" in printed


def test_check_says_so_when_every_patient_has_every_tissue(tmp_path, capsys):
    folder = matched_split(tmp_path / "complete")
    assert main(["check", "--train", str(folder), "--tissues", "2"]) == 0
    printed = capsys.readouterr().out
    assert printed.count("every patient has a sample here") == 2
    assert "padded in" not in printed


def test_check_says_when_validation_repeats_training(counts_folder, capsys):
    train = counts_folder("train", seed=1)
    same = counts_folder("validation", seed=1)  # the same draw
    with pytest.warns(Warning):
        main(["check", "--train", str(train), "--validation", str(same)])
    assert "the validation data is the training data" in capsys.readouterr().out


def test_train_writes_a_model_that_can_be_loaded(counts_folder, tmp_path):
    from preffect.checkpoint import load_checkpoint

    out = tmp_path / "model.pt"
    code = main([
        "train", "--train", str(counts_folder("train")),
        "--out", str(out), "--epochs", "3", "--batch-size", "8",
        "--seed", "0", "--threads", "2",
    ])
    assert code == 0
    assert out.is_file(), "train reported success but wrote nothing"

    checkpoint = load_checkpoint(out)
    assert checkpoint.n_genes == 12
    assert checkpoint.gene_names is not None


def test_train_reports_the_loss_falling(counts_folder, tmp_path, capsys):
    main([
        "train", "--train", str(counts_folder("train")),
        "--out", str(tmp_path / "m.pt"), "--epochs", "20", "--batch-size", "8",
        "--learning-rate", "0.01", "--seed", "0", "--threads", "2",
    ])
    printed = capsys.readouterr().out
    assert "training loss" in printed
    first, last = (
        float(x) for x in printed.split("training loss")[1].split("\n")[0].split("->")
    )
    assert last < first


def test_apply_writes_cleaned_counts(counts_folder, tmp_path):
    import anndata as ad

    train = counts_folder("train")
    model = tmp_path / "model.pt"
    main(["train", "--train", str(train), "--out", str(model),
          "--epochs", "3", "--batch-size", "8", "--seed", "0", "--threads", "2"])

    out = tmp_path / "cleaned.h5ad"
    assert main(["apply", "--model", str(model), "--data", str(train),
                 "--out", str(out), "--threads", "2"]) == 0
    assert out.is_file(), "apply reported success but wrote nothing"

    result = ad.read_h5ad(out)
    assert result.shape == (24, 12)
    assert "composition" in result.layers
    assert result.obsm["latent"].shape[0] == 24


def test_apply_matches_genes_by_name(counts_folder, tmp_path, capsys):
    """Finding 18: a column must mean the same gene it did during training.

    **The columns are deliberately shuffled before applying.** An earlier
    version of this test applied the model to the very folder it trained on, in
    the same column order, and asserted the words "matched 12 genes" appeared.
    Matching by position would have passed it. An audit found that by making
    selection positional and watching this test stay green. Finding 31.
    """
    import anndata as ad

    train = counts_folder("train")
    model = tmp_path / "model.pt"
    main(["train", "--train", str(train), "--out", str(model),
          "--epochs", "2", "--batch-size", "8", "--seed", "0", "--threads", "2"])

    # The same data with its gene columns in a different order. Matching by
    # name must undo this; matching by position cannot.
    source = next(train.glob("*.h5ad"))
    original = ad.read_h5ad(source)
    order = [3, 7, 0, 11, 5, 1, 9, 2, 10, 4, 8, 6]
    shuffled_dir = tmp_path / "shuffled"
    shuffled_dir.mkdir()
    original[:, order].copy().write_h5ad(shuffled_dir / source.name)

    out = tmp_path / "out.h5ad"
    assert main(["apply", "--model", str(model), "--data", str(shuffled_dir),
                 "--out", str(out), "--threads", "2"]) == 0
    assert "matched 12 genes" in capsys.readouterr().out

    # The result must be in the model's gene order, not the file's.
    result = ad.read_h5ad(out)
    assert list(result.var_names) == list(original.var_names)


@pytest.mark.slow
def test_the_single_model_goes_through_the_whole_command_line(counts_folder, tmp_path):
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    train = counts_folder("train", with_network=True)
    model = tmp_path / "single.pt"
    assert main(["train", "--train", str(train), "--out", str(model),
                 "--kind", "single", "--epochs", "3", "--batch-size", "8",
                 "--latent-dim", "4", "--hidden-dim", "12",
                 "--seed", "0", "--threads", "2"]) == 0

    out = tmp_path / "cleaned.h5ad"
    assert main(["apply", "--model", str(model), "--data", str(train),
                 "--out", str(out), "--threads", "2"]) == 0
    assert out.is_file()


# --------------------------------------------------------------------------
# Settings reach the run.
#
# Version 0.1 passed its settings into a parameter that did not exist, so they
# vanished into keyword arguments and the module defaults were used instead.
# --------------------------------------------------------------------------


def test_the_requested_epoch_count_is_what_runs(counts_folder, tmp_path, capsys):
    main(["train", "--train", str(counts_folder("train")),
          "--out", str(tmp_path / "m.pt"), "--epochs", "7",
          "--batch-size", "8", "--threads", "2"])
    assert "finished after 7 epoch(s)" in capsys.readouterr().out


def test_the_requested_shape_is_what_is_built(counts_folder, tmp_path):
    from preffect.checkpoint import load_checkpoint

    out = tmp_path / "m.pt"
    main(["train", "--train", str(counts_folder("train")), "--out", str(out),
          "--epochs", "2", "--batch-size", "8", "--latent-dim", "3",
          "--hidden-dim", "9", "--threads", "2"])
    config = load_checkpoint(out).model_config
    assert config.latent_dim == 3
    assert config.hidden_dim == 9


def test_a_seed_makes_the_whole_command_repeatable(counts_folder, tmp_path):
    from preffect.checkpoint import load_checkpoint

    train = counts_folder("train")
    for name in ("a.pt", "b.pt"):
        main(["train", "--train", str(train), "--out", str(tmp_path / name),
              "--epochs", "4", "--batch-size", "8", "--seed", "11", "--threads", "2"])

    first = load_checkpoint(tmp_path / "a.pt").model_state
    second = load_checkpoint(tmp_path / "b.pt").model_state
    for key in first:
        torch.testing.assert_close(first[key], second[key], msg=f"{key} differs")


def test_gene_selection_is_applied_and_recorded(counts_folder, tmp_path, capsys):
    from preffect.checkpoint import load_checkpoint

    out = tmp_path / "m.pt"
    main(["train", "--train", str(counts_folder("train")), "--out", str(out),
          "--epochs", "2", "--batch-size", "8", "--genes", "5", "--threads", "2"])
    assert "keeping 5 genes" in capsys.readouterr().out

    checkpoint = load_checkpoint(out)
    assert checkpoint.n_genes == 5
    assert len(checkpoint.gene_names) == 5


# --------------------------------------------------------------------------
# Failures are reported, not swallowed.
# --------------------------------------------------------------------------


def test_a_missing_folder_fails_with_a_message(tmp_path, capsys):
    code = main(["check", "--train", str(tmp_path / "absent")])
    assert code == 2
    assert "no such folder" in capsys.readouterr().err


def test_no_command_prints_help_and_fails(capsys):
    assert main([]) == 1
    assert "usage:" in capsys.readouterr().out


def test_version_prints_and_succeeds(capsys):
    assert main(["--version"]) == 0
    assert "preffect 0.2" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Starting from a fitted model. Finding 36.
# --------------------------------------------------------------------------


def test_training_can_start_from_a_saved_model(counts_folder, tmp_path, capsys):
    first = tmp_path / "first.pt"
    folder = counts_folder("train")
    main(["train", "--train", str(folder), "--out", str(first),
          "--epochs", "3", "--batch-size", "8", "--seed", "0", "--threads", "2"])

    second = tmp_path / "second.pt"
    code = main(["train", "--train", str(folder), "--out", str(second),
                 "--epochs", "2", "--batch-size", "8", "--seed", "0",
                 "--threads", "2", "--from-model", str(first)])
    assert code == 0
    assert "starting from" in capsys.readouterr().out
    assert second.is_file()


def test_freezing_a_part_that_does_not_exist_is_refused(counts_folder, tmp_path, capsys):
    first = tmp_path / "first.pt"
    folder = counts_folder("train")
    main(["train", "--train", str(folder), "--out", str(first),
          "--epochs", "2", "--batch-size", "8", "--threads", "2"])

    code = main(["train", "--train", str(folder), "--out", str(tmp_path / "s.pt"),
                 "--epochs", "2", "--batch-size", "8", "--threads", "2",
                 "--from-model", str(first), "--freeze", "encodr"])
    assert code == 2
    assert "no parameter starts with" in capsys.readouterr().err


def test_freezing_without_a_source_model_is_refused(counts_folder, tmp_path, capsys):
    code = main(["train", "--train", str(counts_folder("train")),
                 "--out", str(tmp_path / "m.pt"), "--epochs", "2",
                 "--batch-size", "8", "--threads", "2", "--freeze", "encoder"])
    assert code == 2
    assert "only makes sense with --from-model" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The output path is checked before the work, not after. Issue #17.
#
# Finding 28: an experiment refused its output path after training eight
# seeds, throwing away the run it was meant to protect. Training a real cohort
# is hours.
# --------------------------------------------------------------------------


def test_train_refuses_an_unwritable_output_before_training(counts_folder, tmp_path, capsys):
    """And refuses fast, which is how you can tell it checked first."""
    import time

    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)  # readable, not writable
    try:
        started = time.perf_counter()
        code = main(["train", "--train", str(counts_folder("train")),
                     "--out", str(locked / "model.pt"), "--epochs", "500",
                     "--batch-size", "8", "--threads", "2"])
        took = time.perf_counter() - started
    finally:
        locked.chmod(0o700)

    assert code == 2
    assert "cannot write" in capsys.readouterr().err
    assert took < 10, (
        f"took {took:.1f}s to refuse, so it trained first; 500 epochs would "
        f"not finish that fast"
    )


def test_train_refuses_a_directory_as_the_output(counts_folder, tmp_path, capsys):
    folder = tmp_path / "somewhere"
    folder.mkdir()
    code = main(["train", "--train", str(counts_folder("train")),
                 "--out", str(folder), "--epochs", "2",
                 "--batch-size", "8", "--threads", "2"])
    assert code == 2
    assert "Give a file name" in capsys.readouterr().err


def test_train_creates_a_missing_output_folder(counts_folder, tmp_path):
    """Almost always what was meant, so it is made rather than refused."""
    out = tmp_path / "new" / "deeper" / "model.pt"
    assert main(["train", "--train", str(counts_folder("train")), "--out", str(out),
                 "--epochs", "2", "--batch-size", "8", "--threads", "2"]) == 0
    assert out.is_file()


def test_apply_checks_its_output_before_loading_anything(counts_folder, tmp_path, capsys):
    train = counts_folder("train")
    model = tmp_path / "m.pt"
    main(["train", "--train", str(train), "--out", str(model),
          "--epochs", "2", "--batch-size", "8", "--threads", "2"])

    locked = tmp_path / "ro"
    locked.mkdir(mode=0o500)
    try:
        code = main(["apply", "--model", str(model), "--data", str(train),
                     "--out", str(locked / "out.h5ad"), "--threads", "2"])
    finally:
        locked.chmod(0o700)

    assert code == 2
    err = capsys.readouterr().err
    assert "cannot write" in err
    # It refused before saying it had loaded the model, which is the point.
    assert "loaded a" not in err


def test_the_probe_file_does_not_survive(counts_folder, tmp_path):
    out = tmp_path / "clean" / "model.pt"
    main(["train", "--train", str(counts_folder("train")), "--out", str(out),
          "--epochs", "2", "--batch-size", "8", "--threads", "2"])
    leftovers = [p.name for p in out.parent.iterdir() if "write_test" in p.name]
    assert leftovers == [], f"left a probe file behind: {leftovers}"


# --------------------------------------------------------------------------
# The command line's own documentation. Finding 42.
#
# The command line worked. What described it did not: nine options had no
# help text, the getting-started page told people to run `python -m preffect`
# and there was no __main__ to run, and build_parser's docstring described
# tests and generated documentation that did not exist. Nothing had checked
# the command line's documentation against the command line. These read the
# real parser, so the next gap fails the suite instead of reaching a reader.
# --------------------------------------------------------------------------


def _subcommands() -> dict:
    """Every subcommand's parser by name, read from the real parser."""
    import argparse

    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("the parser has no subcommands")


def _long_options(subparser) -> list[str]:
    """The ``--option`` strings a subcommand takes, ``--help`` excluded."""
    return [
        flag
        for action in subparser._actions
        for flag in action.option_strings
        if flag.startswith("--") and flag != "--help"
    ]


def test_the_parser_still_has_the_three_commands():
    """Guards the two tests below: a walk that found nothing would pass them."""
    assert set(_subcommands()) == {"check", "train", "apply", "gui", "example-data"}
    assert len(_long_options(_subcommands()["train"])) >= 10


@pytest.mark.parametrize("command", ["check", "train", "apply", "gui", "example-data"])
def test_every_option_has_help_text(command):
    bare = [
        flag
        for action in _subcommands()[command]._actions
        for flag in action.option_strings
        if flag.startswith("--") and flag != "--help" and not (action.help or "").strip()
    ]
    assert bare == [], f"`preffect {command} --help` says nothing about {bare}"


def _page_section(command: str) -> str:
    """The part of docs/command-line.md under the heading for one command.

    Checked per section rather than over the whole page: ``--device`` was on
    the page for ``train`` while ``apply`` had none, and a whole-page check
    could not tell.
    """
    page = (Path(__file__).parent.parent / "docs_src" / "command-line.md").read_text()
    marker = f"## `preffect {command}`"
    assert marker in page, f"docs/command-line.md has no section headed {marker}"
    rest = page[page.index(marker) + len(marker):]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


@pytest.mark.parametrize("command", ["check", "train", "apply", "gui", "example-data"])
def test_the_command_line_page_names_every_option(command):
    """docs/command-line.md is written by hand. This is what keeps it honest."""
    section = _page_section(command)
    missing = [
        flag for flag in _long_options(_subcommands()[command]) if f"`{flag}`" not in section
    ]
    assert missing == [], (
        f"the `preffect {command}` section of docs/command-line.md does not mention {missing}"
    )


def test_python_dash_m_runs_the_command():
    """The getting-started page tells people to run this. It has to run."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-P", "-m", "preffect", "--version"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "preffect 0.2" in result.stdout


# --------------------------------------------------------------------------
# apply honours --device. Finding 45.
# --------------------------------------------------------------------------


def test_apply_refuses_a_missing_device_before_loading_anything(
    counts_folder, tmp_path, capsys, monkeypatch
):
    train = counts_folder("train")
    model = tmp_path / "m.pt"
    main(["train", "--train", str(train), "--out", str(model),
          "--epochs", "2", "--batch-size", "8", "--threads", "2"])
    capsys.readouterr()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code = main(["apply", "--model", str(model), "--data", str(train),
                 "--out", str(tmp_path / "out.h5ad"), "--threads", "2", "--device", "cuda"])
    assert code == 2
    captured = capsys.readouterr()
    assert "no usable CUDA" in captured.err
    assert "loaded a" not in captured.out, "it loaded the model before refusing the device"


@pytest.mark.gpu
def test_apply_on_a_gpu_matches_the_cpu(counts_folder, tmp_path, gpu_device):
    import anndata as ad
    import numpy as np

    train = counts_folder("train")
    model = tmp_path / "m.pt"
    main(["train", "--train", str(train), "--out", str(model),
          "--epochs", "3", "--batch-size", "8", "--seed", "0", "--threads", "2"])

    on_cpu, on_gpu = tmp_path / "cpu.h5ad", tmp_path / "gpu.h5ad"
    assert main(["apply", "--model", str(model), "--data", str(train),
                 "--out", str(on_cpu), "--threads", "2"]) == 0
    assert main(["apply", "--model", str(model), "--data", str(train),
                 "--out", str(on_gpu), "--threads", "2", "--device", gpu_device]) == 0

    a, b = ad.read_h5ad(on_cpu), ad.read_h5ad(on_gpu)
    np.testing.assert_allclose(a.X, b.X, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(a.obsm["latent"], b.obsm["latent"], rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------
# train --deterministic. Issue #9.
# --------------------------------------------------------------------------


def test_deterministic_without_a_seed_is_refused_by_the_command_line(
    counts_folder, tmp_path, capsys
):
    code = main(["train", "--train", str(counts_folder("train")),
                 "--out", str(tmp_path / "m.pt"), "--epochs", "2",
                 "--batch-size", "8", "--threads", "2", "--deterministic"])
    assert code == 2
    assert "seed" in capsys.readouterr().err


def test_deterministic_runs_on_the_cpu(counts_folder, tmp_path):
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        out = tmp_path / "m.pt"
        assert main(["train", "--train", str(counts_folder("train")), "--out", str(out),
                     "--epochs", "2", "--batch-size", "8", "--threads", "2",
                     "--seed", "0", "--deterministic"]) == 0
        assert out.is_file()
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])


# --------------------------------------------------------------------------
# gui. Issue #3.
# --------------------------------------------------------------------------


def test_gui_without_streamlit_is_refused_with_the_install_hint(monkeypatch, capsys):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: None)
    assert main(["gui"]) == 2
    assert 'preffect[gui]' in capsys.readouterr().err


# --------------------------------------------------------------------------
# train --patience. Early stopping existed in the settings and in train, and
# the command line printed "stopped early" when it happened, and no flag could
# make it happen. Same shape as --deterministic.
# --------------------------------------------------------------------------


def test_patience_stops_a_run_that_has_stopped_improving(counts_folder, tmp_path, capsys):
    """A min-delta no epoch can beat makes every epoch after the first a
    non-improvement, so with patience 2 the run stops at epoch 3."""
    out = tmp_path / "m.pt"
    assert main(["train", "--train", str(counts_folder("train")), "--out", str(out),
                 "--epochs", "200", "--batch-size", "8", "--threads", "2", "--seed", "0",
                 "--patience", "2", "--min-delta", "1e9"]) == 0
    printed = capsys.readouterr().out
    assert "finished after 3 epoch(s)" in printed
    assert "stopped early, best epoch was 1" in printed
    assert out.is_file()


def test_min_delta_without_patience_is_refused(counts_folder, tmp_path, capsys):
    code = main(["train", "--train", str(counts_folder("train")),
                 "--out", str(tmp_path / "m.pt"), "--epochs", "2", "--batch-size", "8",
                 "--threads", "2", "--min-delta", "0.5"])
    assert code == 2
    assert "only means something with --patience" in capsys.readouterr().err


def test_patience_below_one_is_refused_by_the_settings(counts_folder, tmp_path, capsys):
    code = main(["train", "--train", str(counts_folder("train")),
                 "--out", str(tmp_path / "m.pt"), "--epochs", "2", "--batch-size", "8",
                 "--threads", "2", "--patience", "0"])
    assert code == 2
    assert "at least 1" in capsys.readouterr().err


# --------------------------------------------------------------------------
# --stop-on imputation and --hide (issue #28, finding 50).
# --------------------------------------------------------------------------


def test_train_refuses_to_stop_on_imputation_with_nothing_hidden(counts_folder, tmp_path, capsys):
    code = main([
        "train", "--train", str(counts_folder("train")),
        "--out", str(tmp_path / "m.pt"), "--epochs", "3", "--batch-size", "8",
        "--patience", "2", "--stop-on", "imputation", "--seed", "0", "--threads", "2",
    ])
    assert code == 2
    printed = capsys.readouterr()
    assert "--hide" in printed.err + printed.out


def test_train_can_hide_entries_and_stop_on_their_imputation(counts_folder, tmp_path, capsys):
    code = main([
        "train", "--train", str(counts_folder("train")),
        "--out", str(tmp_path / "m.pt"), "--epochs", "6", "--batch-size", "8",
        "--patience", "2", "--stop-on", "imputation", "--hide", "0.2",
        "--seed", "0", "--threads", "2",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "hiding 20% of the training entries" in printed
    line = next(row for row in printed.splitlines() if "imputation error on hidden entries" in row)
    assert "over" in line and "entries" in line and "left out" in line, line



# --------------------------------------------------------------------------
# --edge-balance reaches the settings (issue #13).
# --------------------------------------------------------------------------


def _saved_settings(path: Path) -> dict:
    """The run's settings as the checkpoint recorded them, read raw.

    ``save_checkpoint`` writes the whole configuration under ``full_config``
    for provenance and ``load_checkpoint`` does not read it back, so the file
    itself is what says how the model was fitted.
    """
    return torch.load(path, map_location="cpu", weights_only=True)["full_config"]


def test_edge_balance_reaches_the_saved_settings(counts_folder, tmp_path):
    """The option is read into LossWeights and travels with the model, so a
    saved model says how its network term was scored."""
    out = tmp_path / "m.pt"
    assert main([
        "train", "--train", str(counts_folder("train")), "--out", str(out),
        "--epochs", "2", "--batch-size", "8", "--threads", "2",
        "--edge-balance", "balanced",
    ]) == 0
    assert _saved_settings(out)["loss"]["edge_balance"] == "balanced"


def test_edge_balance_defaults_to_balanced_on_the_command_line(counts_folder, tmp_path):
    """The command line's default is the package's, so the two cannot drift."""
    out = tmp_path / "m.pt"
    main(["train", "--train", str(counts_folder("train")), "--out", str(out),
          "--epochs", "2", "--batch-size", "8", "--threads", "2"])
    assert _saved_settings(out)["loss"]["edge_balance"] == "balanced"


def test_none_is_still_available_for_the_results_produced_under_it(counts_folder, tmp_path):
    """Every graph-model result before 2026-10-30 was produced under 'none',
    and a run that wants to repeat one asks for it by name."""
    out = tmp_path / "m.pt"
    assert main(["train", "--train", str(counts_folder("train")), "--out", str(out),
                 "--epochs", "2", "--batch-size", "8", "--threads", "2",
                 "--edge-balance", "none"]) == 0
    assert _saved_settings(out)["loss"]["edge_balance"] == "none"


def test_edge_balance_refuses_an_unknown_mode(counts_folder, tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["train", "--train", str(counts_folder("train")),
              "--out", str(tmp_path / "m.pt"), "--edge-balance", "weighted"])
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.slow
def test_the_single_model_trains_under_each_edge_balance(counts_folder, tmp_path, capsys):
    """Both modes go through the whole command line on a model that has a
    network term."""
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    train = counts_folder("train", with_network=True)
    for mode in ("balanced", "none"):
        out = tmp_path / f"{mode}.pt"
        assert main([
            "train", "--train", str(train), "--out", str(out),
            "--kind", "single", "--epochs", "3", "--batch-size", "8",
            "--latent-dim", "4", "--hidden-dim", "12", "--edge-balance", mode,
            "--seed", "0", "--threads", "2",
        ]) == 0
        assert out.is_file()


# --------------------------------------------------------------------------
# The full model goes through the command line.
#
# Finding 55: the README said all three models train from Python and from the
# command line. `preffect train --kind full --tissues 2` read the data, built
# the model, printed its parameter count and stopped at its first batch. This
# file ran the simple and the single model end to end and not the full one.
# --------------------------------------------------------------------------


@pytest.fixture
def two_tissue_folder(tmp_path):
    """A matched pair of tissues on disk, with the table that links them."""
    import pandas as pd

    def _make(name: str, n_samples: int = 16, seed: int = 0):
        names = [[f"tau{t}_{i}" for i in range(n_samples)] for t in (1, 2)]
        table = pd.DataFrame({"tau_1": names[0], "tau_2": names[1]})

        network = np.zeros((n_samples, n_samples), dtype=np.int64)
        for i in range(n_samples):  # a ring, so nothing is isolated
            network[i, (i + 1) % n_samples] = 1
            network[(i + 1) % n_samples, i] = 1

        tissues = []
        for index, own in enumerate(names):
            adata = make_counts(n_samples=n_samples, n_genes=10, seed=seed + index,
                                sample_names=own)
            adata.obsm["sample_sample_adj"] = network
            own_table = table.copy()
            own_table.index = pd.Index(own, dtype=object)
            adata.obsm["sample_association"] = own_table
            tissues.append(adata)
        return write_split(tmp_path / name, tissues)

    return _make


@pytest.mark.slow
def test_the_full_model_goes_through_the_whole_command_line(two_tissue_folder, tmp_path):
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    train = two_tissue_folder("train")
    model = tmp_path / "full.pt"
    assert main(["train", "--train", str(train), "--out", str(model),
                 "--kind", "full", "--tissues", "2", "--epochs", "3",
                 "--batch-size", "8", "--latent-dim", "4", "--hidden-dim", "12",
                 "--seed", "0", "--threads", "2"]) == 0
    assert model.is_file(), "train reported success but wrote nothing"

    out = tmp_path / "cleaned.h5ad"
    assert main(["apply", "--model", str(model), "--data", str(train),
                 "--out", str(out), "--threads", "2"]) == 0
    assert out.is_file()


@pytest.mark.slow
def test_the_full_model_reports_its_loss_falling(two_tissue_folder, tmp_path, capsys):
    """The line finding 55's run never reached."""
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    main(["train", "--train", str(two_tissue_folder("train")),
          "--out", str(tmp_path / "full.pt"), "--kind", "full", "--tissues", "2",
          "--epochs", "12", "--batch-size", "8", "--learning-rate", "0.01",
          "--latent-dim", "4", "--hidden-dim", "12",
          "--seed", "0", "--threads", "2"])
    printed = capsys.readouterr().out
    assert "fitting a full model" in printed
    assert "finished after 12 epoch(s)" in printed
    line = next(row for row in printed.splitlines() if "training loss" in row)
    first, last = (float(v) for v in line.split()[-3::2])
    assert last < first, line


@pytest.mark.slow
def test_a_full_model_keeps_its_tissue_count_in_the_checkpoint(two_tissue_folder, tmp_path):
    """``apply`` reads it from there rather than from ``--tissues``."""
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    from preffect.checkpoint import load_checkpoint

    out = tmp_path / "full.pt"
    main(["train", "--train", str(two_tissue_folder("train")), "--out", str(out),
          "--kind", "full", "--tissues", "2", "--epochs", "2", "--batch-size", "8",
          "--latent-dim", "4", "--hidden-dim", "12", "--seed", "0", "--threads", "2"])
    checkpoint = load_checkpoint(out)
    assert checkpoint.extra["n_tissues"] == 2
    assert checkpoint.build_model().n_tissues == 2


def test_hiding_entries_is_refused_for_the_full_model(two_tissue_folder, tmp_path, capsys):
    """Hiding entries across matched tissues is not built, and says so."""
    pytest.importorskip("torch_geometric", reason="needs the graph extra")

    code = main(["train", "--train", str(two_tissue_folder("train")),
                 "--out", str(tmp_path / "full.pt"), "--kind", "full",
                 "--tissues", "2", "--epochs", "2", "--batch-size", "8",
                 "--latent-dim", "4", "--hidden-dim", "12",
                 "--hide", "0.2", "--seed", "0", "--threads", "2"])
    assert code == 2
    printed = capsys.readouterr()
    assert "one table per tissue" in printed.err + printed.out


# --------------------------------------------------------------------------
# Finding 61: --likelihood chooses the count distribution.
#
# The zero-inflated distribution, the model paths that use it and the loss
# were all ported and tested, and then there was no way to ask for it from
# the command line: `grep likelihood src/preffect/cli.py` found nothing until
# 2026-11-02. Version 0.1 documented `model_likelihood: NB | ZINB` in its
# settings page. Nothing recorded a decision to drop it, so it was an
# omission, and these tests are what stops it being omitted again.
# --------------------------------------------------------------------------


def test_likelihood_reaches_the_saved_settings(counts_folder, tmp_path):
    out = tmp_path / "m.pt"
    assert main([
        "train", "--train", str(counts_folder("train")), "--out", str(out),
        "--epochs", "2", "--batch-size", "8", "--threads", "2",
        "--likelihood", "ZINB",
    ]) == 0
    assert _saved_settings(out)["model"]["likelihood"] == "ZINB"


def test_likelihood_defaults_to_the_negative_binomial(counts_folder, tmp_path):
    """The command line's default is the package's, so the two cannot drift.

    The paper found little evidence that fRNA-seq needs the zero-inflated
    form, which is the reason for this default; a flag that quietly changed it
    would change every result produced without the flag.
    """
    out = tmp_path / "m.pt"
    main(["train", "--train", str(counts_folder("train")), "--out", str(out),
          "--epochs", "2", "--batch-size", "8", "--threads", "2"])
    assert _saved_settings(out)["model"]["likelihood"] == "NB"


def test_likelihood_refuses_an_unknown_distribution(counts_folder, tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["train", "--train", str(counts_folder("train")),
              "--out", str(tmp_path / "m.pt"), "--likelihood", "poisson"])
    assert "invalid choice" in capsys.readouterr().err


def test_a_zero_inflated_model_trains_and_then_applies(counts_folder, tmp_path, capsys):
    """The whole way through, because the distribution is rebuilt on reload.

    `apply` constructs the model from the settings the checkpoint carries. If
    the likelihood did not travel with it, a model fitted as zero-inflated
    would come back as a plain one and predict different counts without
    saying anything.
    """
    out = tmp_path / "zinb.pt"
    assert main([
        "train", "--train", str(counts_folder("train")), "--out", str(out),
        "--epochs", "3", "--batch-size", "8", "--seed", "0", "--threads", "2",
        "--likelihood", "ZINB",
    ]) == 0

    import anndata as ad

    cleaned = tmp_path / "cleaned.h5ad"
    assert main(["apply", "--model", str(out), "--data", str(counts_folder("other", seed=1)),
                 "--out", str(cleaned), "--threads", "2"]) == 0
    assert cleaned.is_file(), "apply reported success but wrote nothing"
    assert ad.read_h5ad(cleaned).shape == (24, 12)
    assert "matched 12 genes" in capsys.readouterr().out


def test_the_saved_model_rebuilds_with_the_distribution_it_was_fitted_with(
    counts_folder, tmp_path
):
    from preffect.checkpoint import load_checkpoint

    out = tmp_path / "zinb.pt"
    main(["train", "--train", str(counts_folder("train")), "--out", str(out),
          "--epochs", "2", "--batch-size", "8", "--seed", "0", "--threads", "2",
          "--likelihood", "ZINB"])
    checkpoint = load_checkpoint(out)
    assert checkpoint.model_config.likelihood.value == "ZINB"
    assert checkpoint.build_model().config.likelihood.value == "ZINB"


def test_asking_for_zero_inflation_builds_the_layer_that_produces_the_zeros(
    counts_folder, tmp_path, capsys
):
    """Otherwise the flag could be read, saved and never used, and every test
    above would still pass. Version 0.1's habit was exactly this: honour a
    setting in the record and not in the run.

    The zero-inflated model needs a head the plain one has no use for, one
    that produces the log-odds of an extra zero per gene and sample. Its
    presence in the saved weights is what says the flag reached the model.
    """
    folder = counts_folder("train")
    paths = {}
    for likelihood in ("NB", "ZINB"):
        paths[likelihood] = tmp_path / f"{likelihood}.pt"
        main(["train", "--train", str(folder), "--out", str(paths[likelihood]),
              "--epochs", "3", "--batch-size", "8", "--seed", "0", "--threads", "2",
              "--likelihood", likelihood])
    printed = capsys.readouterr().out

    states = {
        name: torch.load(path, map_location="cpu", weights_only=True)["model_state"]
        for name, path in paths.items()
    }
    extra = set(states["ZINB"]) - set(states["NB"])
    assert extra, "the zero-inflated model has no layer the plain one lacks"
    assert all("zero" in key for key in extra), extra
    assert not set(states["NB"]) - set(states["ZINB"])

    # And the run said so: the parameter count the command line prints differs.
    assert "7,514 parameters" in printed
    assert "8,126 parameters" in printed


# --------------------------------------------------------------------------
# `preffect example-data`: the worked example needs data a tool install can
# make, since an install from GitHub has no vignettes/ folder to run a script
# from.
# --------------------------------------------------------------------------


def test_example_data_writes_every_kind_and_split(tmp_path, capsys):
    out = tmp_path / "example"
    assert main(["example-data", "--out", str(out)]) == 0
    for kind in ("simple", "single", "full"):
        for split in ("train", "validation", "test"):
            assert (out / kind / split / "counts.tau_1.h5ad").is_file(), f"{kind}/{split}"
    assert (out / "full" / "train" / "counts.tau_2.h5ad").is_file()
    printed = capsys.readouterr().out
    assert "No sample appears in more than one split" in printed
    assert "md5" in printed


def test_example_data_can_be_checked_and_fitted(tmp_path, capsys):
    """The data it writes is usable by the other commands, end to end."""
    out = tmp_path / "example"
    assert main(["example-data", "--out", str(out)]) == 0
    assert main(["check", "--train", str(out / "simple" / "train"),
                 "--validation", str(out / "simple" / "validation")]) == 0
    assert "0 of 120 training samples reappear" in capsys.readouterr().out
    assert main(["train", "--train", str(out / "simple" / "train"),
                 "--out", str(tmp_path / "m.pt"), "--epochs", "2", "--seed", "0"]) == 0
    assert (tmp_path / "m.pt").is_file()


def test_example_data_refuses_an_unmatched_share_out_of_range(tmp_path, capsys):
    assert main(["example-data", "--out", str(tmp_path / "x"), "--unmatched", "1.0"]) == 2
    assert "unmatched must be at least 0 and below 1" in capsys.readouterr().err
