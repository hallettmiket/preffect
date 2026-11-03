"""The command line: prepare data, check it, fit a model, apply it.

Three commands do the work, and the first one is the one to run first::

    preffect check   --train <folder>            # does my data look usable?
    preffect train   --train <folder> --out <file>
    preffect apply   --model <file> --data <folder> --out <file>
    preffect gui                                  # the same, in a browser
    preffect example-data --out example_data      # a small simulated cohort to try them on

Two more come before them, for anyone whose counts are not in PREFFECT's
format yet::

    preffect prepare --counts counts.csv --samples samples.csv --out my_study/train
    preffect network --data my_study/train --out my_study/train_linked

**Version 0.1's command line did nothing.** It read its arguments, built a
session object and exited without calling any task, so a run appeared to
succeed and produced no output. It also passed its settings into a parameter
that did not exist, so they were silently ignored and the module-level defaults
were used instead. That is finding 11, and the remedy is not cleverness but a
test that runs each command end to end and checks a file comes out.

Every command prints what it is doing and what it found, because a long run
that says nothing is indistinguishable from one that has hung.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``preffect`` command.

    Separate from :func:`main` so that the command line can be inspected
    without running anything. Two tests read it: one fails if any option has
    no help text, and one fails if ``docs/command-line.md`` does not mention
    every option a command takes. So the page and the ``--help`` output can
    only drift apart with the suite saying so. An earlier version of this
    sentence described documentation generated from the parser; none was, and
    nothing read the parser either. Finding 42.
    """
    parser = argparse.ArgumentParser(
        prog="preffect",
        description="Model RNA sequencing counts from FFPE material.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    subcommands = parser.add_subparsers(dest="command")

    check = subcommands.add_parser(
        "check",
        help="read the data and report whether it can be used",
        description=(
            "Reads the counts without fitting anything, reports what it found, "
            "and says plainly if something would stop a run. Worth doing first."
        ),
    )
    check.add_argument("--train", required=True, type=Path, help="folder of training counts")
    check.add_argument("--validation", type=Path, help="folder of validation counts, optional")
    check.add_argument("--tissues", type=int, default=1, help="how many tissues (default 1)")

    train = subcommands.add_parser(
        "train", help="fit a model and save it", description="Fit a model to counts."
    )
    train.add_argument("--train", required=True, type=Path, help="folder of training counts")
    train.add_argument("--validation", type=Path, help="folder of validation counts, optional")
    train.add_argument("--out", required=True, type=Path, help="where to write the model")
    train.add_argument(
        "--tissues",
        type=int,
        default=1,
        help=(
            "how many tissue files each folder holds, one per matched tissue "
            "(default 1). Only the full model uses more than one."
        ),
    )
    train.add_argument(
        "--kind",
        default="simple",
        choices=["simple", "single", "full"],
        help=(
            "simple uses counts alone; single adds one sample-sample network; "
            "full uses several matched tissues per patient (default simple)"
        ),
    )
    train.add_argument(
        "--likelihood",
        default="NB",
        choices=["NB", "ZINB"],
        help=(
            "which count distribution the model fits: NB, the negative binomial "
            "(default), or ZINB, which adds a separate probability that a count "
            "is zero for a reason unrelated to expression level. The paper found "
            "little evidence that fRNA-seq needs ZINB, which is why NB is the "
            "default; `preffect.zinb_fit` asks the question of your own counts"
        ),
    )
    train.add_argument(
        "--epochs", type=int, default=100, help="passes over the training data (default 100)"
    )
    train.add_argument(
        "--batch-size", type=int, default=100, help="samples per step (default 100)"
    )
    train.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="how far the optimiser moves per step (default 0.001)",
    )
    train.add_argument(
        "--patience",
        type=int,
        help=(
            "stop early once the validation loss, or the training loss when there "
            "is no --validation, has not improved for this many epochs; the model "
            "kept is the best epoch's (default: run every epoch)"
        ),
    )
    train.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="how much the loss must fall to count as improving, for --patience (default 0)",
    )
    train.add_argument(
        "--stop-on",
        choices=["loss", "imputation"],
        default="loss",
        help=(
            "what --patience watches: the loss (default; the validation loss, or the "
            "training loss without --validation), or imputation, the error on the "
            "entries --hide hid, for a model that imputes the cohort it is fitted on"
        ),
    )
    train.add_argument(
        "--hide",
        type=float,
        default=None,
        metavar="FRACTION",
        help=(
            "hide this fraction of the training entries at random (0.2 hides a fifth); "
            "the model is scored against the full counts, the error on the hidden "
            "entries is reported at the end, and --stop-on imputation watches it"
        ),
    )
    train.add_argument(
        "--edge-balance",
        choices=["balanced", "none"],
        default="balanced",
        help=(
            "how the single and full models score the pairs of samples that are "
            "not linked, which are 98 to 99.8 per cent of a real network: balanced "
            "gives the linked pairs the same total weight as the absent ones "
            "(default); none scores every pair equally, as version 0.1 did, under "
            "which the network half learns nothing"
        ),
    )
    train.add_argument(
        "--latent-dim",
        type=int,
        default=20,
        help="size of the summary learned for each sample (default 20)",
    )
    train.add_argument(
        "--hidden-dim",
        type=int,
        default=50,
        help="width of the hidden layers (default 50)",
    )
    train.add_argument("--genes", type=int, help="keep only this many of the most variable genes")
    train.add_argument("--seed", type=int, help="set for a repeatable run")
    train.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "repeatable bit for bit on a GPU as well as on the CPU; needs --seed. "
            "Without it, two GPU runs of a graph model from the same seed differ "
            "at the fourth decimal place."
        ),
    )
    train.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "compute threads (default 4). PyTorch otherwise uses one per core, "
            "which for this work is far slower, not faster. See finding 24."
        ),
    )
    train.add_argument(
        "--device", default="cpu", help="cpu, or cuda, or cuda:0 (default cpu)"
    )
    train.add_argument(
        "--from-model",
        type=Path,
        help=(
            "start from a model already fitted, instead of from scratch. Needs "
            "the same genes and the same shape; it says which if not."
        ),
    )
    train.add_argument(
        "--freeze",
        action="append",
        default=[],
        metavar="PART",
        help=(
            "hold part of the model still, for example --freeze encoder. "
            "Repeatable. Only with --from-model. A part that does not exist is "
            "refused rather than freezing nothing, which is what version 0.1 "
            "did. See finding 36."
        ),
    )

    apply_ = subcommands.add_parser(
        "apply",
        help="run a saved model over data and write the cleaned counts",
        description="Apply a fitted model to counts and write the result.",
    )
    apply_.add_argument("--model", required=True, type=Path, help="a saved model")
    apply_.add_argument("--data", required=True, type=Path, help="folder of counts")
    apply_.add_argument("--out", required=True, type=Path, help="where to write the result")
    apply_.add_argument(
        "--tissues",
        type=int,
        default=1,
        help=(
            "how many tissue files the data folder holds (default 1). Only "
            "consulted if the model did not record its own count; a model "
            "saved by this version always does."
        ),
    )
    apply_.add_argument(
        "--threads", type=int, default=4, help="compute threads (default 4), as for train"
    )
    apply_.add_argument(
        "--device", default="cpu", help="cpu, or cuda, or cuda:0 (default cpu), as for train"
    )

    gui = subcommands.add_parser(
        "gui",
        help="open the graphical interface in a browser",
        description=(
            "Opens PREFFECT's graphical interface in your browser: check a folder "
            "of counts, apply a saved model, and look at the before-and-after "
            "figures. Runs on this machine only; nothing leaves it."
        ),
    )
    gui.add_argument(
        "--port", type=int, default=8501, help="the local port to serve on (default 8501)"
    )

    example = subcommands.add_parser(
        "example-data",
        help="write a small simulated cohort to try the other commands on",
        description=(
            "Writes simulated counts with a group signal in them: three splits with "
            "no sample in common, for the simple model, the single model (with a "
            "sample-sample network that has real edges) and the full model (two "
            "tissues per patient). About two seconds, 200 genes."
        ),
    )
    example.add_argument(
        "--out", type=Path, default=Path("example_data"),
        help="folder to write under (default example_data)",
    )
    example.add_argument(
        "--seed", type=int, default=0,
        help="seed for the simulation (default 0); the same seed writes the same files",
    )
    example.add_argument(
        "--unmatched", type=float, default=0.0,
        help="share of patients whose second tissue is left out of the full model's "
        "data, so there is something to pad; at least 0 and below 1 (default 0, "
        "which writes the same files as before the option existed)",
    )

    # The default number of neighbours comes from the preparation module so
    # that the measured number lives in one place rather than two.
    from preffect.prepare import DEFAULT_NEIGHBOURS, ORIENTATIONS

    prepare = subcommands.add_parser(
        "prepare",
        help="build tissue files from a counts table and a sample table",
        description=(
            "Turns a table of counts and a table describing the samples into "
            "the files the other commands read. Give --counts and --samples "
            "once per tissue, in tissue order."
        ),
    )
    prepare.add_argument(
        "--counts", required=True, action="append", type=Path, metavar="FILE",
        help=(
            "a comma- or tab-separated table of counts, whole numbers, with the "
            "sample names in the first column and the gene names in the first "
            "row. Give it once per tissue, in tissue order."
        ),
    )
    prepare.add_argument(
        "--samples", required=True, action="append", type=Path, metavar="FILE",
        help=(
            "a comma- or tab-separated table describing those samples, one row "
            "each, with the sample names in the first column. Once per tissue, "
            "in the same order as --counts."
        ),
    )
    prepare.add_argument(
        "--out", required=True, type=Path, help="folder to write the tissue files in"
    )
    prepare.add_argument(
        "--counts-are",
        default=ORIENTATIONS[0],
        choices=list(ORIENTATIONS),
        help=(
            "which way round the counts table is written (default "
            f"{ORIENTATIONS[0]}, meaning one row per sample). Use "
            f"{ORIENTATIONS[1]} for a table with one row per gene, which is how "
            "many pipelines write one; it is turned the right way round here."
        ),
    )
    prepare.add_argument(
        "--sample-column",
        help="which column of the sample table holds the sample names (default: the first)",
    )
    prepare.add_argument(
        "--name", default="counts",
        help="what to call the files before the tissue number (default counts, "
             "giving counts.tau_1.h5ad)",
    )
    prepare.add_argument(
        "--patient-column",
        help=(
            "which column of the sample tables says which patient a sample came "
            "from. Required with more than one tissue, because that column is "
            "what says a sample in one tissue and a sample in another are the "
            "same patient; the full model cannot run without it"
        ),
    )
    prepare.add_argument(
        "--neighbours", type=int, metavar="N",
        help=(
            "also build a sample-sample network, linking each sample to the N "
            "others whose counts look most like its own. Needed by the single "
            f"and full models; {DEFAULT_NEIGHBOURS} is the number this project "
            "measured. Left out, no network is written and only the simple "
            "model applies"
        ),
    )
    prepare.add_argument(
        "--mutual", action="store_true",
        help="with --neighbours, keep a link only where each of the two samples "
             "is among the other's N most similar, which gives a sparser network",
    )
    prepare.add_argument(
        "--label", metavar="COLUMN",
        help="a column of the sample table, such as a subtype, to measure the "
             "network against: the summary then says what share of the links "
             "join two samples with the same value of it, and what share would "
             "by chance",
    )
    prepare.add_argument(
        "--overwrite", action="store_true",
        help="write over files that are already in --out (default: refuse)",
    )

    network = subcommands.add_parser(
        "network",
        help="add a sample-sample network to counts that have none",
        description=(
            "Builds a sample-sample network from the counts themselves and "
            "writes a copy of every tissue file carrying it. The files read are "
            "left exactly as they are. The single and full models need this; "
            "the simple model does not."
        ),
    )
    network.add_argument("--data", required=True, type=Path, help="folder of counts to read")
    network.add_argument(
        "--out", required=True, type=Path,
        help="folder to write the copies in, under the same file names",
    )
    network.add_argument(
        "--tissues", type=int, default=1,
        help="how many tissue files the folder holds (default 1)",
    )
    network.add_argument(
        "--neighbours", type=int, default=DEFAULT_NEIGHBOURS, metavar="N",
        help=(
            f"link each sample to the N others whose counts look most like its "
            f"own (default {DEFAULT_NEIGHBOURS}, the number measured on this "
            f"project's own cohort). A link is kept wherever either of the two "
            f"names the other"
        ),
    )
    network.add_argument(
        "--mutual", action="store_true",
        help="keep a link only where each of the two samples is among the "
             "other's N most similar, which gives a sparser network",
    )
    network.add_argument(
        "--label", metavar="COLUMN",
        help="a sample annotation, such as a subtype, to measure the network "
             "against: the summary then says what share of the links join two "
             "samples with the same value of it, and what share would by chance",
    )
    network.add_argument(
        "--replace", action="store_true",
        help=(
            "build a new network for a file that already carries one with links "
            "in it (default: refuse, and say so). A network entry holding no "
            "links at all, which is what every dataset version 0.1 shipped "
            "carries, is replaced either way"
        ),
    )
    network.add_argument(
        "--overwrite", action="store_true",
        help="write over files that are already in --out (default: refuse)",
    )
    return parser



def check_writable(path: Path, what: str) -> None:
    """Fail now rather than after the work, if this path cannot be written.

    Training a real cohort runs for hours. Discovering at the end that the
    output folder does not exist, or is read-only, or is a directory where a
    file was meant, throws the whole run away. Finding 28 is that happening:
    an experiment refused its output path after training eight seeds.

    Creates the parent folder, since that is almost always what was meant, and
    writes and removes a probe file so permissions are tested rather than
    assumed.
    """
    from preffect.errors import PreffectError

    if path.is_dir():
        raise PreffectError(
            f"{what} is {path}, which is a directory. Give a file name."
        )
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PreffectError(
            f"cannot create the folder for {what} at {parent}: {error}"
        ) from error

    probe = parent / f".preffect_write_test_{os.getpid()}"
    try:
        probe.touch()
        probe.unlink()
    except OSError as error:
        raise PreffectError(
            f"cannot write {what} to {parent}: {error}. Checked before "
            f"starting, because a run that fails at the end has thrown away "
            f"everything it did."
        ) from error


def command_check(args) -> int:
    from preffect.data.checks import warn_if_validation_repeats_training
    from preffect.data.loading import load_tissues
    from preffect.data.tissues import unmatched_counts
    from preffect.example_data import TRUTH_LAYER
    from preffect.workflow import describe_network

    train_tissues = load_tissues(args.train, args.tissues)
    print(f"training: {args.tissues} tissue(s) from {args.train}")
    for index, tissue in enumerate(train_tissues, start=1):
        print(
            f"  tissue {index}: {tissue.n_obs} samples, {tissue.n_vars} genes, "
            f"{describe_network(tissue)}"
        )
        # Only simulated data can carry the count an entry was drawn from, so
        # finding it is worth saying out loud: it tells a reader the file is
        # not measured, and it is what an imputation example scores against.
        if TRUTH_LAYER in tissue.layers:
            print(
                f"  tissue {index}: carries the count each entry was drawn from in "
                f"layers['{TRUTH_LAYER}'], so these counts are simulated"
            )
    # How many patients lack a sample in each tissue. The full model pads a row
    # of per-gene means in for each of them, so this is the line that says how
    # much of the cohort it will be working on is made up rather than measured.
    for index, count in enumerate(unmatched_counts(train_tissues).values(), start=1):
        if count:
            print(
                f"  tissue {index}: {count} patient(s) have no sample here, and "
                f"get a row of per-gene means padded in"
            )
        else:
            print(f"  tissue {index}: every patient has a sample here")

    from preffect.data.checks import check_tissues

    check_tissues(train_tissues, split="training")
    print("  the counts look usable")

    if args.validation:
        validation = load_tissues(args.validation, args.tissues)
        check_tissues(validation, split="validation")
        print(f"validation: {validation[0].n_obs} samples from {args.validation}")
        overlaps = warn_if_validation_repeats_training(train_tissues, validation)
        for index, overlap in enumerate(overlaps, start=1):
            if overlap.is_effectively_identical:
                print(
                    f"  tissue {index}: the validation data is the training data. "
                    f"Leave --validation out instead."
                )
            else:
                print(
                    f"  tissue {index}: {overlap.n_identical_rows} of "
                    f"{overlap.n_train} training samples reappear in validation"
                )
    else:
        print("validation: none given, so no validation loss will be reported")
    return 0


def command_train(args) -> int:
    import time

    from preffect.checkpoint import (
        TRAINING_HISTORY_KEY,
        save_checkpoint,
        training_history_record,
    )
    from preffect.config import (
        DataConfig,
        Likelihood,
        LossWeights,
        ModelConfig,
        ModelKind,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.data.selection import apply_gene_selection, choose_genes
    from preffect.errors import ConfigError
    from preffect.training import configure_runtime
    from preffect.training import train as fit
    from preffect.workflow import build_model, read_split, to_model_data

    if args.min_delta != 0.0 and args.patience is None:
        raise ConfigError(
            "--min-delta only means something with --patience; without it no run "
            "stops early and the value would be ignored"
        )
    if args.stop_on == "imputation" and args.patience is None:
        raise ConfigError(
            "--stop-on imputation only means something with --patience; without it "
            "no run stops early and the choice would be ignored"
        )
    if args.stop_on == "imputation" and args.hide is None:
        raise ConfigError(
            "--stop-on imputation watches the entries --hide hides, and nothing is "
            "hidden; add --hide 0.2, or stop on the loss"
        )
    if args.hide is not None and not 0 < args.hide < 1:
        raise ConfigError(f"--hide is a fraction strictly between 0 and 1, got {args.hide}")

    kind = ModelKind(args.kind)
    model_config = ModelConfig(
        kind=kind,
        likelihood=Likelihood(args.likelihood),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )
    config = PreffectConfig(
        data=DataConfig(
            train_path=args.train,
            validation_path=args.validation,
            n_tissues=args.tissues,
            select_genes=args.genes,
        ),
        model=model_config,
        loss=LossWeights(edge_balance=args.edge_balance).for_tissue_count(args.tissues),
        train=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.patience,
            early_stopping_min_delta=args.min_delta,
            stop_on=args.stop_on,
        ),
        runtime=RuntimeConfig(
            seed=args.seed,
            device=args.device,
            num_threads=args.threads,
            deterministic=args.deterministic,
        ),
    )

    # Before anything is built, and before the data is read. A training run on
    # a real cohort is hours; finding out then that the output cannot be
    # written loses all of it. Issue #17.
    check_writable(args.out, "--out")

    # Before anything is built. Weights are initialised when a model is
    # constructed, so seeding after that leaves the initialisation unseeded and
    # the run unrepeatable. Found by a test that ran the same command twice.
    configure_runtime(config)

    needs_network = kind.uses_adjacency
    tissues = read_split(args.train, args.tissues)

    if args.genes is not None:
        selection = choose_genes(tissues[0], args.genes)
        print(f"keeping {len(selection)} genes, chosen from the training data")
        tissues = apply_gene_selection(tissues, selection, "training")

    gene_names = tuple(str(n) for n in tissues[0].var_names)
    n_genes = len(gene_names)
    train_data = to_model_data(tissues, model_config, needs_network)
    if args.hide is not None:
        from preffect.workflow import hide_entries

        train_data = hide_entries(train_data, args.hide, args.seed)
        print(f"hiding {args.hide:.0%} of the training entries; the model is scored "
              f"against the full counts")

    validation_data = None
    if args.validation:
        validation_tissues = read_split(args.validation, args.tissues)
        if args.genes is not None:
            # The same genes, matched by name, never re-chosen. Finding 18.
            validation_tissues = apply_gene_selection(
                validation_tissues, selection, "validation"
            )
        validation_data = to_model_data(validation_tissues, model_config, needs_network)

    model = build_model(kind, n_genes, model_config, args.tissues)

    if getattr(args, "from_model", None) is not None:
        from preffect.pretraining import load_pretrained

        record = load_pretrained(model, args.from_model, freeze=tuple(args.freeze))
        print(f"starting from {args.from_model}: {record}")
    elif getattr(args, "freeze", None):
        from preffect.errors import ConfigError

        raise ConfigError(
            "--freeze only makes sense with --from-model; there is nothing to "
            "hold still in a model that has not been fitted yet"
        )

    print(
        f"fitting a {args.kind} model: {n_genes} genes, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters, "
        f"{args.epochs} epochs"
    )

    # Timed here rather than inside ``train``, so the figure is the fitting
    # alone: reading the data and matching the genes happened above, and a
    # reader comparing two runs wants the part that scales with the epochs.
    started = time.perf_counter()
    history = fit(model, train_data, config, validation_data)
    fit_seconds = time.perf_counter() - started
    last = history.epochs[-1]
    print(f"finished after {len(history)} epoch(s)")
    print(f"  training loss   {history.train_losses[0]:.2f} -> {last.train.mean_loss:.2f}")
    if last.validation is not None:
        print(f"  validation loss {last.validation.mean_loss:.2f}")
    if history.stopped_early:
        print(f"  stopped early, best epoch was {history.best_epoch}")
    if args.hide is not None:
        from preffect.training import hidden_entry_scores

        # Measured on the model being saved, which after early stopping is the
        # best epoch's rather than the last. The count says what the error was
        # computed over; entries whose true value is zero are left out of it.
        scores = hidden_entry_scores(model, train_data)
        left_out = scores["zero_truth_entries_skipped"]
        print(f"  imputation error on hidden entries {scores['hidden']:.3f}, over "
              f"{scores['n_hidden'] - left_out:,} entries "
              f"({left_out:,} hidden entries with a true value of zero left out)")

    # The run's own account of how it went travels with the model, so that a
    # file opened weeks later says how many epochs it took, whether it was
    # stopped early and how long it ran. Without it the loss curve exists only
    # in the terminal that printed it.
    record = training_history_record(
        history,
        epochs_asked_for=args.epochs,
        wall_seconds=fit_seconds,
        patience=args.patience,
    )
    save_checkpoint(
        args.out,
        model,
        model_config,
        n_genes,
        gene_names=gene_names,
        full_config=config,
        n_tissues=args.tissues,
        **{TRAINING_HISTORY_KEY: record},
    )
    print(f"wrote {args.out}")
    return 0


def command_apply(args) -> int:
    import torch

    from preffect.config import RuntimeConfig
    from preffect.training import resolve_device
    from preffect.workflow import apply_checkpoint

    torch.set_num_threads(args.threads)
    check_writable(args.out, "--out")
    # Refused by name before the model is loaded or the data read, for the same
    # reason the output path is: a device found missing after the loading has
    # wasted the loading. Finding 45.
    device = resolve_device(RuntimeConfig(device=args.device, num_threads=args.threads))

    result, tissues, _ = apply_checkpoint(
        args.model, args.data, n_tissues=args.tissues, device=device, report=print
    )
    target = tissues[0]
    adata = result.to_anndata(
        sample_names=[str(n) for n in target.obs_names],
        gene_names=[str(n) for n in target.var_names],
    )
    adata.write_h5ad(args.out)
    print(f"wrote {args.out}: {adata.n_obs} samples, {adata.n_vars} genes")
    return 0


def command_gui(args) -> int:
    """Start the graphical interface, or say what to install.

    Replaces this process with Streamlit's rather than starting it as a child:
    one process, so stopping ``preffect gui`` stops the server. The first
    version spawned a child, and killing the parent left a server running on
    a shared machine with nobody attached to it.
    """
    import importlib.util

    from preffect.errors import PreffectError

    if importlib.util.find_spec("streamlit") is None:
        raise PreffectError(
            "the graphical interface needs streamlit, which is an optional "
            'dependency. Install it with: pip install "preffect[gui]"'
        )
    page = Path(__file__).with_name("gui.py")
    print(f"opening the PREFFECT interface at http://localhost:{args.port}")
    print("it answers on this machine only; press Ctrl+C here to stop it")
    command = [
        sys.executable, "-m", "streamlit", "run", str(page),
        "--server.port", str(args.port),
        # Only this machine. Streamlit's default is every network interface,
        # which on a shared server would offer a page pointed at governed data
        # to anyone on the network. From a laptop, forward the port over SSH.
        "--server.address", "127.0.0.1",
        # Streamlit reports usage to its makers unless told not to. This
        # interface is pointed at governed data, so it is told not to.
        "--browser.gatherUsageStats", "false",
    ]
    sys.stdout.flush()
    os.execv(sys.executable, command)
    return 0  # pragma: no cover - execv does not return


def command_example_data(args) -> int:
    from preffect.example_data import write_example_data

    written = write_example_data(args.out, seed=args.seed, unmatched=args.unmatched)
    print(written.describe())
    return 0


def _refuse_existing(targets: list[Path], overwrite: bool) -> None:
    """Stop before any file is written if one of them is there already.

    Checked for the whole folder up front rather than file by file, so a
    two-tissue run cannot write its first file and then refuse its second,
    leaving a folder that is half old and half new.
    """
    from preffect.errors import PreffectError

    if overwrite:
        return
    existing = [t for t in targets if t.exists()]
    if existing:
        raise PreffectError(
            f"{len(existing)} file(s) are already in that folder: "
            f"{', '.join(p.name for p in existing[:3])}"
            + (f" and {len(existing) - 3} more" if len(existing) > 3 else "")
            + ". Writing over them would lose whatever they hold. Give another "
            "--out, or pass --overwrite if replacing them is what you meant."
        )


def command_network(args) -> int:
    from preffect.data.loading import find_tissue_files
    from preffect.prepare import add_network_to_file

    sources = find_tissue_files(args.data, args.tissues)
    targets = [args.out / source.name for source in sources]
    _refuse_existing(targets, args.overwrite)

    for index, (source, target) in enumerate(zip(sources, targets, strict=True), start=1):
        print(f"tissue {index}: {source}")
        written = add_network_to_file(
            source,
            target,
            neighbours=args.neighbours,
            mutual=args.mutual,
            label_column=args.label,
            replace=args.replace,
            overwrite=args.overwrite,
        )
        for line in written.describe().splitlines():
            print(f"  {line}")
    print(
        f"the counts are unchanged; fit a graph model on them with "
        f"preffect train --train {args.out} --kind single"
    )
    return 0


def command_prepare(args) -> int:
    from preffect.errors import ConfigError
    from preffect.prepare import (
        correspondence_table,
        describe_sample_network,
        nearest_neighbour_network,
        read_counts_table,
        read_sample_table,
        tissue_file_name,
        write_tissue,
    )

    if len(args.counts) != len(args.samples):
        raise ConfigError(
            f"{len(args.counts)} counts table(s) and {len(args.samples)} sample "
            f"table(s) were given; there has to be one of each per tissue, in the "
            f"same order"
        )
    # Refused by name rather than ignored, as --min-delta is for --patience:
    # an option that silently does nothing is how version 0.1 ran a model
    # nobody asked for.
    if args.neighbours is None and (args.label or args.mutual):
        named = "--label" if args.label else "--mutual"
        raise ConfigError(
            f"{named} describes the sample-sample network, and --neighbours was "
            f"not given, so no network is built and the setting would be ignored. "
            f"Add --neighbours, or leave {named} out."
        )
    n_tissues = len(args.counts)
    if n_tissues > 1 and args.patient_column is None:
        raise ConfigError(
            f"{n_tissues} tissues were given and --patient-column was not. The "
            f"files share no sample names, so that column is the only thing that "
            f"says a sample in one tissue and a sample in another came from the "
            f"same patient, and the full model cannot be fitted without it."
        )

    targets = [args.out / tissue_file_name(i, args.name) for i in range(1, n_tissues + 1)]
    _refuse_existing(targets, args.overwrite)

    counts_per_tissue = []
    samples_per_tissue = []
    for index, (counts_path, samples_path) in enumerate(
        zip(args.counts, args.samples, strict=True), start=1
    ):
        counts = read_counts_table(counts_path, args.counts_are)
        samples = read_sample_table(samples_path, args.sample_column)
        if args.label and args.label not in samples.columns:
            raise ConfigError(
                f"tissue {index}'s sample table has no column called "
                f"{args.label!r} to measure the network against. It has: "
                f"{list(samples.columns) or '(nothing but the sample names)'}"
            )
        print(
            f"tissue {index}: {counts.shape[0]} samples and {counts.shape[1]} "
            f"genes from {counts_path}, read as {args.counts_are}"
        )
        print(f"  {samples.shape[1]} annotation(s) per sample from {samples_path}")
        counts_per_tissue.append(counts)
        samples_per_tissue.append(samples)

    association = None
    if n_tissues > 1:
        association = correspondence_table(samples_per_tissue, args.patient_column)
        print(f"{len(association)} patient(s) across {n_tissues} tissues")
        for index in range(1, n_tissues + 1):
            unmatched = int((association[f"tau_{index}"] == "NA").sum())
            if unmatched:
                print(
                    f"  tissue {index}: {unmatched} patient(s) have no sample "
                    f"here, and the full model pads a row of per-gene means in "
                    f"for each of them"
                )
            else:
                print(f"  tissue {index}: every patient has a sample here")

    for index, (counts, samples) in enumerate(
        zip(counts_per_tissue, samples_per_tissue, strict=True), start=1
    ):
        network = None
        if args.neighbours is not None:
            network = nearest_neighbour_network(
                counts.to_numpy(), args.neighbours, mutual=args.mutual
            )
            labels = samples[args.label].to_numpy() if args.label else None
            summary = describe_sample_network(network, labels, args.label or "label")
            for line in summary.describe().splitlines():
                print(f"  tissue {index}: {line}")
        path = write_tissue(
            args.out,
            counts,
            samples,
            tissue=index,
            stem=args.name,
            network=network,
            association=association,
            overwrite=args.overwrite,
        )
        print(f"wrote {path}")

    print(f"check it with: preffect check --train {args.out}" + (
        f" --tissues {n_tissues}" if n_tissues > 1 else ""
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``preffect`` command and return the exit status.

    This is what the installed ``preffect`` command calls. Errors raised by the
    package are caught here and printed as a sentence rather than a traceback,
    because a traceback tells someone who did not write the code nothing they
    can act on. Anything else is left to propagate, since an unexpected
    exception is a bug and the traceback is the useful part of it.

    :param argv: arguments to parse. Left unset it reads ``sys.argv``, which is
        what happens from a terminal; pass a list to call it from a test or
        from another program.
    :returns: 0 if the command did what was asked, non-zero otherwise. Suitable
        for ``raise SystemExit(main())``.
    """
    from preffect import __version__
    from preffect.errors import PreffectError

    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        print(f"preffect {__version__}")
        return 0
    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "check": command_check,
        "train": command_train,
        "apply": command_apply,
        "gui": command_gui,
        "example-data": command_example_data,
        "prepare": command_prepare,
        "network": command_network,
    }
    try:
        return handlers[args.command](args)
    except PreffectError as error:
        print(f"preffect: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
