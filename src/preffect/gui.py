"""The graphical interface: open a run, read how it fitted, then look at it page by page.

Started with ``preffect gui``, which runs this file under Streamlit and opens
it in a browser. It runs on this machine only and sends nothing anywhere;
Streamlit's own usage reporting is switched off by the command that starts it.

It covers the part of the work a person wants to *see*: turning two tables
into the file the rest of the package reads, whether a folder of counts can
be fitted and what it holds, how the fit that produced a saved model went,
the figures that say whether to believe it, what the model made of one gene
or one sample, whether it can fill in a count it was not shown, and how two
saved models compare on the same counts. Fitting a model is hours on a real
cohort and belongs on the command line, where it can be left running; the
interface reads what ``preffect train`` wrote.

## The pages are grouped by what they need

The sidebar lists the pages under six headings, in the order the work is
done: *Start*, where a run is opened; *Your data*, the three pages that need
no model; *The fit*, which describes the saved model; *Does the fit hold?*,
one page per figure; *Ask the model*, the two pages that put a question to
the open run; and *Compare*, once a second model is open. A page is listed
only when what it needs is open, so a page is never offered with nothing
behind it, and every page that reports something is titled with the question
it answers. The first version offered one flat list of radio buttons, with the
page about the fitted model first, the two steps that come before any model
in the middle and the figures after them, so a reader could not tell which
pages were about their data and which were about a model, and landed on an
empty page. Finding 67.

## Why Streamlit's own navigation, and why the pages are functions

Streamlit's multi-page mechanism once wanted a ``pages/`` directory beside the
script, one file per page, which is why the first version drew a radio list
instead: this package ships one module that ``preffect gui`` runs by path.
Since Streamlit 1.36 a page can be a plain function, so :func:`render` builds
the list from what is open, hands it to ``st.navigation``, and each page is a
closure over the open run. One entry point, headings in the sidebar, and an
address per page that can be bookmarked.

## Why the form is on the landing page, and how its values survive

The folder, the model and the device are asked for once, on *Start here*,
rather than in a sidebar repeated on every page. Streamlit forgets a widget's
value as soon as a run passes without drawing it, which is every visit to
another page, so :func:`render` writes each form value back into the session
on every run, which is the documented way to keep it. The run itself, once
opened, is held in the session under its own key and is not a widget.

## The one fit the interface makes: the example

The interface does not fit models, because a real cohort takes hours and
belongs on the command line. The one exception is the button on the landing
page that writes the simulated example cohort and fits the worked example's
two models on it, which takes seconds, by calling the command line's own
``train`` in this process with the worked example's exact settings. It fills
the form with the paths it made, under the system's temporary folder, so a
newcomer sees every page populated without typing a path and nothing in the
package names one.

## Why it reads as a report rather than one long page

The first version drew every figure one after another, so a reader met six
pictures with nothing between them and no way to tell, before looking, whether
the fit behind them had finished moving. Now the run is described first, with
its loss curve, and each figure gets a page of its own that opens by saying
what the picture shows, what a good one looks like and what to do about a bad
one. Only the chosen page is drawn, so changing the colouring of the latent
summary no longer redraws five figures that did not change.

## One description and one drawing call per figure

A figure page and the comparison of two runs draw the same figures, so the
description of each lives in :data:`FIGURE_DESCRIPTIONS` and the call that
draws it in :func:`_draw_figure`, once. Writing them out twice is how the
page a reader sees and the page they compare against drift apart.

Every figure of the data is one of :mod:`preffect.plots`' figures, drawn by the
same call a script would make, so a picture seen here can be reproduced and put
in a paper. The loss curve on the fit page is the exception and is drawn here:
the figure in :mod:`preffect.plots` needs every term of the objective at every
epoch, and what a checkpoint carries is the two totals.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["render"]

#: The pages, each named by the question it answers, and the one page that
#: writes a file named by the instruction. Every title is a constant so that
#: the tests, the comparison page and the documentation cannot drift from the
#: sidebar.
START_PAGE = "Start here"
PREPARE_PAGE = "Prepare the input file"
CHECK_PAGE = "Can this be fitted?"
LOOK_PAGE = "What does the data look like?"
FIT_PAGE = "How did the fit go?"
DEPTH_PAGE = "Does it track sequencing depth?"
COUNTS_PAGE = "Does it reproduce the counts?"
BATCH_PAGE = "Did it remove the batch effect?"
NOISE_PAGE = "Is the noise per gene believable?"
ZEROS_PAGE = "Does it get the zeros right?"
GROUPS_PAGE = "Do similar samples group together?"
LOOKUP_PAGE = "What did it make of one gene or sample?"
IMPUTE_PAGE = "Can it fill in hidden counts?"
COMPARE_PAGE = "How do two models compare?"

#: The pages that need no model, in the order they are done: write the file,
#: find out whether it can be fitted, then look at what it holds.
DATA_PAGES = (PREPARE_PAGE, CHECK_PAGE, LOOK_PAGE)

#: The figure pages, in the order a reader should meet them: the simplest
#: check of the fit first, the latent summary last.
FIGURE_PAGES = (DEPTH_PAGE, COUNTS_PAGE, BATCH_PAGE, NOISE_PAGE, ZEROS_PAGE, GROUPS_PAGE)

#: Pages that put a question to the open run rather than describing it.
TOOL_PAGES = (LOOKUP_PAGE, IMPUTE_PAGE)

#: The sidebar, top to bottom: each heading and the pages under it, in the
#: order the work is done. A page is listed only when what it needs is open.
SECTIONS = (
    ("Start", (START_PAGE,)),
    ("Your data", DATA_PAGES),
    ("The fit", (FIT_PAGE,)),
    ("Does the fit hold?", FIGURE_PAGES),
    ("Ask the model", TOOL_PAGES),
    ("Compare", (COMPARE_PAGE,)),
)

#: Each page's address, so that a page can be bookmarked and returned to.
URL_PATHS = {
    START_PAGE: "start",
    PREPARE_PAGE: "prepare",
    CHECK_PAGE: "check",
    LOOK_PAGE: "look",
    FIT_PAGE: "fit",
    DEPTH_PAGE: "depth",
    COUNTS_PAGE: "counts",
    BATCH_PAGE: "batch",
    NOISE_PAGE: "noise",
    ZEROS_PAGE: "zeros",
    GROUPS_PAGE: "groups",
    LOOKUP_PAGE: "lookup",
    IMPUTE_PAGE: "impute",
    COMPARE_PAGE: "compare",
}

#: The form on the landing page, by widget key. Written back into the session
#: on every run, so that leaving the page does not blank them.
FORM_KEYS = ("data_folder", "n_tissues", "model_path", "device", "second_model_path")

#: The worked example's own fitting recipe (`docs_src/vignette.md`), which is
#: what the example button runs, through the command line's ``train``.
EXAMPLE_RECIPE = ("--epochs", "80", "--batch-size", "40", "--threads", "4", "--patience", "5")

#: What each figure shows, what a good one looks like, what a bad one looks
#: like, and what to do about it. Written for a reader who has not read the
#: code. Held here rather than inside each page because the comparison of two
#: runs shows the same figures and must say the same thing about them.
FIGURE_DESCRIPTIONS = {
    DEPTH_PAGE: (
        "The library size of a sample is the total of all its counts, and it "
        "says how deeply that sample was sequenced rather than anything "
        "biological. The left panel is the observed total for each sample and "
        "the right is the total the model inferred, both drawn on the same "
        "scale so that two different pictures cannot look like the same one. If "
        "the model has learned the depth of sequencing, the two histograms sit "
        "in the same place and are about as wide as each other. If the inferred "
        "one is shifted sideways or much narrower, the model is not tracking "
        "depth, and every count it predicts carries that error. Before blaming "
        "the model, check that the observed totals are not themselves extreme: "
        "one sample sequenced ten times as deeply as the rest will pull the "
        "fit, and dropping it is usually the right answer."
    ),
    COUNTS_PAGE: (
        "This is the plainest question to ask of a fitted model: does it "
        "reproduce the counts it was given. Each panel is one gene, with the "
        "observed count for a sample across and the count the model fitted for "
        "that same sample up, and the dashed line is where the two agree. The "
        "genes shown are the ones that vary most across samples, because a gene "
        "that barely moves is easy to fit and tells you nothing. A good result "
        "has the points following the dashed line over the whole range, in "
        "every panel. Points squashed into a horizontal band mean the model is "
        "predicting that gene's average for every sample and has learned "
        "nothing about which sample is which, which is usually too few epochs "
        "or a latent space that collapsed during training."
    ),
    BATCH_PAGE: (
        "A batch is a group of samples handled together, and a batch effect is "
        "a difference between those groups that came from the handling rather "
        "than from the biology. Each row of the figure compares two batches: "
        "the average count of every gene in the first batch across, the same "
        "gene's average in the second up, and the dashed line is where the two "
        "agree. The left panel uses the observed counts and the right uses the "
        "model's fitted counts. Correction has worked when the right panel "
        "hugs the line more closely than the left, meaning a difference that "
        "was in the data is no longer in the model's view of it. If the two "
        "panels look alike, the model was not told about the batches: name the "
        "column as a correction variable and fit again."
    ),
    NOISE_PAGE: (
        "Four panels, and the last is the one worth your time. Top left is each "
        "gene's average count, observed across and fitted up; top right is how "
        "the fitted dispersion is spread across genes; bottom left is each "
        "sample's total count, observed against fitted. Dispersion is how much "
        "more variable a gene's counts are than pure counting noise would make "
        "them, so a large value means an erratic gene. The bottom right panel "
        "puts each gene's average count against its dispersion, both on a "
        "logarithmic scale, and in real data the dispersion falls as the "
        "average rises. A flat or rising cloud there means the dispersion is "
        "not being fitted, and the model's idea of how noisy each gene is "
        "cannot be trusted; fit for longer, and check the loss curve on the fit "
        "page before reading anything else here."
    ),
    ZEROS_PAGE: (
        "A count of zero means one of two things: the gene was genuinely off in "
        "that sample, or it was there and was missed. The zero-inflated model "
        "carries a separate probability that any given count is a zero for the "
        "second reason, and this page asks whether that probability matches "
        "what the data did. Across is the fraction of samples in which a gene "
        "was observed to be zero, up is the average probability of a zero the "
        "model gives it, and the dashed line is where the two agree. A good "
        "result has the points along that line over the whole range. Points "
        "well above the line mean the model expects more zeros than there are, "
        "which inflates everything else it predicts for those genes, and points "
        "well below mean it is explaining real zeros as ordinary low counts "
        "instead."
    ),
    GROUPS_PAGE: (
        "The latent summary is the model's own short description of each "
        "sample: a few dozen numbers it decides are enough to rebuild that "
        "sample's counts. This page squeezes those numbers down to two so they "
        "can be drawn, one point per sample. The two axes have no units and no "
        "meaning of their own, so only the arrangement of the points says "
        "anything. Colour the points by a column of your sample table and the "
        "question becomes whether samples sharing a label sit together: groups "
        "you expect to differ sitting apart is the good result, and it is the "
        "evidence that the model captured something real. One undifferentiated "
        "cloud means either that the model learned nothing about which sample "
        "is which, or that the column you coloured by is not what drives this "
        "data, so colour by something else before concluding the first."
    ),
}

#: How much two runs have to differ on a measure before the comparison table
#: names a winner. **Chosen for this table, not measured.** One fit of each
#: model cannot say how much of a difference is the seed, which is why the
#: project's own experiments use six seeds; these thresholds exist only to
#: stop a difference too small to act on being read as a result.
DECIDABLE_DIFFERENCE = {
    "median correlation, fitted against observed": 0.01,
    "latent clustering against the chosen column (adjusted Rand)": 0.05,
}

#: The same, as a relative difference, for a measure whose scale depends on
#: the cohort rather than running from zero to one.
DECIDABLE_RELATIVE_DIFFERENCE = 0.05


@dataclass(frozen=True)
class Run:
    """One opened run: a saved model, the counts it was applied to, the result.

    Held as one object because the comparison page works with two of them and
    every page needs the same five things out of each.
    """

    name: str
    result: Any
    tissues: Any
    checkpoint: Any

    @property
    def target(self):
        """The tissue whose counts the model predicts, genes already matched."""
        return self.tissues[0]

    @property
    def observed(self):
        """The counts the model was given, dense, whatever the file stored."""
        return _as_dense(self.target.X)

    @property
    def gene_names(self) -> list[str]:
        return [str(n) for n in self.target.var_names]


def _as_dense(values):
    """Counts as a dense array, whatever the file stored them as."""
    import numpy as np

    return np.asarray(values.toarray() if hasattr(values, "toarray") else values)


def _colour_columns(obs, limit: int = 20) -> list[str]:
    """Columns of the sample table that look like labels: few distinct values."""
    return [c for c in obs.columns if obs[c].nunique(dropna=False) <= limit]


def _group_columns(obs, limit: int = 20) -> list[str]:
    """Label columns with at least two groups, so a comparison is possible."""
    return [c for c in _colour_columns(obs, limit) if obs[c].nunique(dropna=False) >= 2]


def _duration(seconds: float) -> str:
    """A length of time a person can read at a glance."""
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _setting(full_config, section: str, name: str, fallback: str = "not recorded"):
    """One setting out of the configuration a checkpoint carried.

    Older files were written without the configuration, and a run that was
    fitted from Python may not have passed one. Say so rather than showing a
    default that nothing chose, which is how version 0.1's shipped paths
    reached a published figure.
    """
    if not isinstance(full_config, dict):
        return fallback
    value = full_config.get(section, {})
    if not isinstance(value, dict) or name not in value:
        return fallback
    return value[name]


def _loss_figure(record):
    """The objective over the epochs the run took, from what it recorded.

    One line per split, on shared axes, because the gap between training and
    validation is the thing worth seeing and two figures hide it. The dotted
    line marks the epoch whose weights were saved, drawn only after an early
    stop, where it is somewhere other than the end.
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 3.6))
    epochs = list(record.epochs)
    axis.plot(epochs[: len(record.train_loss)], record.train_loss,
              label="training", linewidth=2)
    if record.validation_loss:
        axis.plot(epochs[: len(record.validation_loss)], record.validation_loss,
                  label="validation", linewidth=2)
    if record.stopped_early and record.best_epoch is not None:
        axis.axvline(record.best_epoch, linestyle=":", color="grey", linewidth=1)
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean loss per batch")
    axis.legend(fontsize=9)
    figure.tight_layout()
    return figure


def _show(st, plt, draw) -> None:
    """Draw one figure and close it, reporting a refusal as a sentence.

    A figure that cannot be drawn from this model says why on the page, in the
    sentence :mod:`preffect.plots` refuses with. Closing matters because
    Streamlit redraws on every interaction and matplotlib keeps every figure
    it is handed until something closes it.
    """
    from preffect.errors import PreffectError

    try:
        figure = draw()
    except PreffectError as error:
        st.info(str(error))
        return
    st.pyplot(figure)
    plt.close(figure)


def _match_axes(first, second) -> None:
    """Give two figures of the same shape one set of limits, panel by panel.

    Two pictures on different scales look like two different results even
    when they are the same one, which is the mistake this exists to prevent.
    Panels are paired in the order the figures created them, and a figure with
    fewer panels than the other simply stops early rather than failing.
    """
    for left, right in zip(first.axes, second.axes, strict=False):
        for read, write in (("get_xlim", "set_xlim"), ("get_ylim", "set_ylim")):
            low = min(getattr(left, read)()[0], getattr(right, read)()[0])
            high = max(getattr(left, read)()[1], getattr(right, read)()[1])
            getattr(left, write)(low, high)
            getattr(right, write)(low, high)


def _draw_figure(plots, page: str, run: Run, choices: dict):
    """The figure one figure page shows, for this run.

    One place, so that a figure page and the comparison of two runs cannot
    draw different things under the same heading.

    :param choices: what the page's own controls selected, if any: the column
        holding the batch, and the column to colour the latent summary by.
    :raises PreffectError: in the sentence :mod:`preffect.plots` refuses with,
        when this model has nothing to draw for this page.
    """
    result, observed = run.result, run.observed
    if page == DEPTH_PAGE:
        return plots.library_size_histogram(result, observed)
    if page == COUNTS_PAGE:
        return plots.gene_scatter(result, observed, gene_names=run.gene_names)
    if page == BATCH_PAGE:
        batches = run.target.obs[choices["batch_column"]].astype(str).to_numpy()
        return plots.batch_adjustment(result, observed, batches)
    if page == NOISE_PAGE:
        return plots.library_size_and_dispersion(result, observed)
    if page == ZEROS_PAGE:
        return plots.zero_fraction_against_pi(result, observed)
    if page == GROUPS_PAGE:
        colour = choices["colour_by"]
        values = None if colour == "none" else run.target.obs[colour].astype(str).to_numpy()
        return plots.latent_umap(result, colour_by=values, colour_label=colour)
    raise KeyError(page)


def _figure_controls(st, page: str, run: Run, key_prefix: str) -> dict | None:
    """Whatever a figure page has to be told before it can draw.

    Returns ``None`` when the page cannot be drawn at all for this run and has
    already said so, which is a batch comparison on a cohort whose sample
    table names no groups.
    """
    choices = {"batch_column": None, "colour_by": "none"}
    if page == BATCH_PAGE:
        columns = _group_columns(run.target.obs)
        if not columns:
            st.info(
                "No column of the sample table names a group with at least two "
                "values, so there are no batches to compare. A batch column holds "
                "one label per sample, such as the run or the site it came from."
            )
            return None
        default = columns.index("batch") if "batch" in columns else 0
        choices["batch_column"] = st.selectbox(
            "Column holding the batch", columns, index=default, key=f"{key_prefix}batch_by"
        )
    if page == GROUPS_PAGE:
        choices["colour_by"] = st.selectbox(
            "Colour by", ["none"] + _colour_columns(run.target.obs), key=f"{key_prefix}colour_by"
        )
    return choices


# ---------------------------------------------------------------------------
# The pages. Each opens by saying what it is for, before any picture.
# ---------------------------------------------------------------------------


def _check_page(st, data_folder: str, n_tissues: int) -> None:
    """Whether a folder of counts could be fitted at all, without fitting it."""
    from preffect.workflow import inspect_split

    st.header(CHECK_PAGE)
    st.markdown(
        "This is what `preffect check` reports, and it needs no model. It says "
        "how many samples and genes each tissue file holds, whether there is a "
        "sample-sample network and how many links it has, and, if something "
        "would stop a run, the sentence the run would have refused with. Run it "
        "on a new folder before you spend hours fitting anything. It is a pass "
        "or a fail; the page after it says what the folder holds."
    )
    if not data_folder:
        st.info(f"Give a folder of counts on {START_PAGE}, then press the button here.")
        return
    if st.button("Check the folder", key="check"):
        st.session_state["report"] = inspect_split(Path(data_folder), n_tissues)

    report = st.session_state.get("report")
    if report is not None:
        _show_report(st, report)


def _show_report(st, report) -> None:
    """What ``preffect check`` found, as a table and the sentence it ends on.

    Shared by the page that checks a folder and the page that writes one, so
    that preparing data finishes on exactly the sentence checking it would
    have given.
    """
    st.caption(str(report.folder))
    if report.tissues:
        st.table(
            [
                {"tissue": i, "samples": t.samples, "genes": t.genes, "network": t.network}
                for i, t in enumerate(report.tissues, start=1)
            ]
        )
    if report.usable:
        st.success("The counts look usable.")
    else:
        st.error(report.problem)


def _look_page(st, plt, plots, data_folder: str, n_tissues: int) -> None:
    """What a folder of counts holds, read without a model, for the choices before a fit."""
    from preffect.errors import PreffectError

    st.header(LOOK_PAGE)
    st.markdown(
        "What is in a folder of counts, read without fitting anything, for the "
        "decisions that come before a fit. For each tissue file it says how "
        "many samples and genes there are and how much of the matrix is zero, "
        "which is what decides whether to ask `preffect train` for the "
        "zero-inflated distribution; what columns the sample table has and how "
        "many values each takes, which is where a batch to correct for or a "
        "label to colour by comes from; whether the file carries a "
        "sample-sample network, which only the graph models use; and how deeply "
        "each sample was sequenced, as a picture. A sample sequenced ten times "
        "as deeply as the rest, or a library that failed, shows up here before "
        "any model has been pulled by it, and this is the moment to drop it. "
        "The page before this one says whether the folder can be fitted at "
        "all; this one says what you would be fitting."
    )
    if not data_folder:
        st.info(f"Give a folder of counts on {START_PAGE}, then press the button here.")
        return
    if st.button("Read the folder", key="look"):
        try:
            with st.spinner("Reading the counts"):
                st.session_state["looked"] = _read_for_a_look(Path(data_folder), n_tissues)
        except PreffectError as error:
            st.session_state.pop("looked", None)
            st.error(str(error))

    looked = st.session_state.get("looked")
    if looked is None:
        return
    if looked["folder"] != str(Path(data_folder)):
        st.info(
            f"This is {looked['folder']}, which is no longer the folder on "
            f"{START_PAGE}. Press the button again to read {data_folder}."
        )
        return
    st.caption(looked["folder"])
    several = len(looked["tissues"]) > 1
    for index, tissue in enumerate(looked["tissues"], start=1):
        if several:
            st.subheader(f"Tissue {index}")
        _describe_tissue(st, plt, plots, tissue)


def _read_for_a_look(folder: Path, n_tissues: int) -> dict:
    """Read the tissue files as they are, without the checks a run would make.

    The loader refuses a file it cannot read, in its own sentence. Whether the
    counts would stop a run is the check page's question, and a folder that
    would fail it is still worth looking at, which is why this reads and does
    not check.
    """
    from preffect.data.loading import load_tissues

    return {"folder": str(folder), "tissues": load_tissues(folder, n_tissues)}


def _describe_tissue(st, plt, plots, tissue) -> None:
    """One tissue file: its size, its zeros, its sample table, its network, its depth."""
    import numpy as np

    from preffect.data.graph import has_adjacency
    from preffect.workflow import describe_network

    observed = _as_dense(tissue.X)
    totals = observed.sum(axis=1)
    zeros = observed == 0
    row = st.columns(4)
    row[0].metric("samples", tissue.n_obs)
    row[1].metric("genes", tissue.n_vars)
    row[2].metric(
        "entries that are zero",
        f"{float(zeros.mean()):.0%}" if zeros.size else "none",
        help="The share of all the counts in the matrix that are zero. A matrix "
        "that is largely zeros is where the zero-inflated distribution, "
        "`preffect train --likelihood ZINB`, earns its extra parameter; one "
        "that is not usually does not need it.",
    )
    row[3].metric(
        "median library size",
        f"{float(np.median(totals)):,.0f}" if totals.size else "none",
        help="A sample's library size is the total of its counts, which says "
        "how deeply it was sequenced.",
    )
    if zeros.size:
        per_gene = zeros.mean(axis=0)
        st.caption(
            f"The median gene is zero in {float(np.median(per_gene)):.0%} of "
            f"samples, and {int((per_gene == 1.0).sum())} gene(s) are zero in "
            f"every sample."
        )

    st.markdown("**The sample table**")
    obs = tissue.obs
    if obs.shape[1] == 0:
        st.write(
            "No columns: nothing is known about the samples beyond their "
            "counts, so there is no batch to correct for and nothing to colour "
            "the latent summary by."
        )
    else:
        st.table([
            {
                "column": str(column),
                "distinct values": int(obs[column].nunique(dropna=False)),
                "missing": int(obs[column].isna().sum()),
                "for example": ", ".join(
                    str(v) for v in obs[column].dropna().drop_duplicates().head(3)
                ),
            }
            for column in obs.columns
        ])
        st.caption(
            "A column with a few distinct values, such as the run or the site a "
            "sample came from, is a batch: the model can be told about it as a "
            "correction variable when it is fitted from Python, the batch page "
            "compares the groups it names, and the latent summary can be "
            "coloured by any column with up to twenty values."
        )

    st.markdown("**The sample-sample network**")
    if has_adjacency(tissue):
        st.write(f"The file carries {describe_network(tissue)}.")
    else:
        st.write(
            "None in the file. The simple model needs none; the single and full "
            "models do, and `preffect network` adds one."
        )

    st.markdown("**How deeply each sample was sequenced**")
    _show(st, plt, lambda: plots.observed_library_sizes(observed))
    st.caption(
        "Each sample's total count. A bar far to the right of the rest is a "
        "sample sequenced far more deeply than the others and one far to the "
        "left is a failed library; either pulls a fit towards itself, and "
        "dropping it before fitting is usually the right answer."
    )


def _prepare_page(st) -> None:
    """Turn a counts table and a sample table into the file everything else reads."""
    from preffect.errors import PreffectError
    from preffect.prepare import ORIENTATIONS

    st.header(PREPARE_PAGE)
    st.markdown(
        "PREFFECT reads one file per tissue, and this page writes it from the "
        "two tables you are likely to have: a counts table with a count for "
        "every sample and gene, and a sample table saying what is known about "
        "each sample. The file it writes is an `.h5ad`, named "
        "`counts.tau_1.h5ad`, holding the counts, the annotations, and "
        "optionally a sample-sample network and the table that says which "
        "sample of one tissue is which patient in another. Nothing is written "
        "until you press the button at the bottom, nothing already in the "
        "folder is written over, and the page finishes by running the same "
        "check `preffect check` runs, so you end on the sentence that says "
        "whether what you made can be fitted."
    )

    n_tissues = int(st.number_input(
        "Tissues to prepare", min_value=1, max_value=10, value=1, key="prepare_tissues",
        help="One is the common case. Several matched profiles per patient is "
        "what the full model takes.",
    ))
    source = st.radio(
        "Where the tables are",
        ("Upload them", "Already on this machine"),
        key="prepare_source",
        horizontal=True,
        help="Typing a path is usually right when PREFFECT runs on a server "
        "and you reached this page over an SSH tunnel: the file is already "
        "there, and uploading sends it through the browser for nothing.",
    )
    orientation = st.radio(
        "The counts table is written", ORIENTATIONS, key="prepare_orientation",
        horizontal=True,
    )
    st.caption(
        f"{ORIENTATIONS[0]} means one row per sample, with the sample names "
        f"down the first column and the gene names across the top. Pick "
        f"{ORIENTATIONS[1]} for a table written the other way round, which is "
        f"how many pipelines write one, and it is flipped as it is read. The "
        f"sample table always has one row per sample, with the names in its "
        f"first column."
    )

    given = _prepare_inputs(st, n_tissues, source)
    if given is None:
        st.info("Give a counts table and a sample table for every tissue.")
        return
    try:
        read = _read_prepared_tables(given, orientation)
    except PreffectError as error:
        st.error(str(error))
        return

    st.subheader("What was read")
    st.table([
        {
            "tissue": index,
            "samples": counts.shape[0],
            "genes": counts.shape[1],
            "annotations per sample": samples.shape[1],
        }
        for index, (counts, samples) in enumerate(read, start=1)
    ])
    st.caption(
        f"Read as {orientation}. The sample table's columns: "
        + "; ".join(
            f"tissue {i}: " + (", ".join(str(c) for c in s.columns) or "none")
            for i, (_, s) in enumerate(read, start=1)
        )
    )

    network_per_tissue = _prepare_network(st, read)
    association = _prepare_correspondence(st, read)
    _prepare_write(st, read, network_per_tissue, association)


def _prepare_inputs(st, n_tissues: int, source: str):
    """One counts table and one sample table per tissue, or ``None`` until both.

    Returns whatever :mod:`preffect.prepare`'s readers accept: the buffer a
    browser upload arrives in, or a path typed in.

    **Every tissue's boxes are drawn before any of them is read.** Stopping at
    the first empty one leaves the second tissue with nowhere to put its
    tables, so a two-tissue cohort can never be filled in at all.
    """
    given = []
    for tissue in range(1, n_tissues + 1):
        if n_tissues > 1:
            st.markdown(f"**Tissue {tissue}**")
        row = st.columns(2)
        if source == "Upload them":
            counts = row[0].file_uploader(
                f"Counts table, tissue {tissue}", key=f"prepare_counts_{tissue}",
                type=["csv", "tsv", "tab", "txt", "gz"],
            )
            samples = row[1].file_uploader(
                f"Sample table, tissue {tissue}", key=f"prepare_samples_{tissue}",
                type=["csv", "tsv", "tab", "txt", "gz"],
            )
        else:
            counts = row[0].text_input(
                f"Counts table, tissue {tissue}", key=f"prepare_counts_path_{tissue}"
            )
            samples = row[1].text_input(
                f"Sample table, tissue {tissue}", key=f"prepare_samples_path_{tissue}"
            )
        given.append((counts, samples))
    if any(not counts or not samples for counts, samples in given):
        return None
    return given


def _read_prepared_tables(given, orientation: str):
    """Read every tissue's two tables, refusing in the reader's own words.

    An uploaded file is rewound first. Streamlit hands back the same buffer on
    every rerun of the script, and a buffer already read to the end reads as
    an empty table the second time the page is touched.
    """
    from preffect.prepare import read_counts_table, read_sample_table

    def rewound(source):
        if hasattr(source, "seek"):
            source.seek(0)
        return source

    return [
        (read_counts_table(rewound(counts), orientation), read_sample_table(rewound(samples)))
        for counts, samples in given
    ]


def _shared_columns(read) -> list[str]:
    """Columns every tissue's sample table has, which are the ones to choose from."""
    if not read:
        return []
    shared = set(str(c) for c in read[0][1].columns)
    for _, samples in read[1:]:
        shared &= {str(c) for c in samples.columns}
    return sorted(shared)


def _prepare_network(st, read) -> list:
    """Optionally build a sample-sample network per tissue, and describe it.

    The summary is redrawn as the slider moves, because the number of
    neighbours is the one choice here with no right answer and seeing what it
    does to the links per sample is how it gets made.
    """
    from preffect.errors import PreffectError
    from preffect.prepare import (
        DEFAULT_NEIGHBOURS,
        describe_sample_network,
        nearest_neighbour_network,
    )

    st.subheader("A sample-sample network")
    st.markdown(
        "Optional, and only the single and full models use one. It links each "
        "sample to the samples whose counts look most like its own, by the "
        "recipe this project measured. If your sample table has a column the "
        "links ought to follow, such as a subtype, name it below: the share of "
        "links joining two samples with that same label, set against the share "
        "that would join by chance, is what says the network carries biology "
        "rather than noise. Two shares close together mean it does not."
    )
    if not st.checkbox("Build a network", key="prepare_network"):
        return [None] * len(read)

    # Two below the smallest cohort, because the builder refuses a count that
    # links every sample to every other and a complete network says nothing
    # about which samples resemble which. It refuses again when the links it
    # built come out complete anyway, and that sentence is shown as it is.
    smallest = min(counts.shape[0] for counts, _ in read)
    most = max(1, min(50, smallest - 2))
    row = st.columns(3)
    neighbours = int(row[0].slider(
        "Neighbours each sample names", min_value=1, max_value=most,
        value=min(DEFAULT_NEIGHBOURS, most), key="prepare_neighbours",
        help="Ten is the number this project's own cohort was measured at. A "
        "link is kept wherever either sample names the other, so most samples "
        "end up with more links than this. The most the slider offers is set "
        "by the smallest tissue, since linking every sample to every other "
        "says nothing.",
    ))
    mutual = row[1].checkbox(
        "Only where each names the other", key="prepare_mutual",
        help="A sparser network, in which some samples end up with no links.",
    )
    choices = ["none", *_shared_columns(read)]
    label = row[2].selectbox("Measure the links against", choices, key="prepare_label")

    networks = []
    for index, (counts, samples) in enumerate(read, start=1):
        try:
            network = nearest_neighbour_network(
                counts.to_numpy(), neighbours, mutual=mutual
            )
            summary = describe_sample_network(
                network,
                None if label == "none" else samples[label].to_numpy(),
                "none" if label == "none" else label,
            )
        except PreffectError as error:
            st.error(f"tissue {index}: {error}")
            networks.append(None)
            continue
        networks.append(network)
        st.markdown(f"**Tissue {index}.** " + summary.describe().replace("\n", "  \n"))
    return networks


def _prepare_correspondence(st, read):
    """Optionally build the table saying which sample is which patient.

    Offered only for two tissues or more. With one tissue there is nothing to
    correspond to, and the full model, which is what reads this table, takes
    at least two.
    """
    from preffect.errors import PreffectError
    from preffect.prepare import MISSING_COUNTERPART, correspondence_table

    if len(read) < 2:
        return None

    st.subheader("Which sample is which patient")
    st.markdown(
        "The full model reads several profiles from the same patient together, "
        "so it has to know which row of one tissue's file is which row of "
        "another's. The files share no sample names, here and usually "
        "elsewhere, so the correspondence is written down instead, from the "
        "one thing the tissues do share: a column naming the patient each "
        "sample came from. A patient with no sample in a tissue is kept, and "
        "the model pads a row of that tissue's per-gene means in for them."
    )
    choices = _shared_columns(read)
    if not choices:
        st.error(
            "No column appears in every tissue's sample table, so there is "
            "nothing that could say two samples are the same patient."
        )
        return None
    column = st.selectbox("Column naming the patient", choices, key="prepare_patient")
    try:
        association = correspondence_table([samples for _, samples in read], column)
    except PreffectError as error:
        st.error(str(error))
        return None

    st.write(f"{len(association)} patient(s) across {len(read)} tissues.")
    st.table([
        {
            "tissue": index,
            "patients with a sample here": int(
                (association[f"tau_{index}"] != MISSING_COUNTERPART).sum()
            ),
            "patients with none": int(
                (association[f"tau_{index}"] == MISSING_COUNTERPART).sum()
            ),
        }
        for index in range(1, len(read) + 1)
    ])
    return association


def _prepare_write(st, read, networks, association) -> None:
    """Write one file per tissue, then check them the way the command line does."""
    from preffect.errors import PreffectError
    from preffect.prepare import write_tissue
    from preffect.workflow import inspect_split

    st.subheader("Write the files")
    st.markdown(
        "Name a folder and the files go in it, one per tissue. **Nothing "
        "already there is written over**: a name that is taken is refused, "
        "with the path it would have replaced, so prepare into a new folder or "
        "move the old one aside. Every count is checked before anything is "
        "written, and a problem names the entries at fault."
    )
    row = st.columns(2)
    folder = row[0].text_input("Folder to write to", key="prepare_out")
    stem = row[1].text_input(
        "Name before the tissue number", value="counts", key="prepare_stem",
        help="counts gives counts.tau_1.h5ad, which is what the loader looks for.",
    )
    if st.button("Write the files and check them", key="prepare_write"):
        if not folder:
            st.error("Name a folder to write the files to.")
        else:
            try:
                written = [
                    str(write_tissue(
                        Path(folder), counts, samples,
                        tissue=index, stem=stem,
                        network=networks[index - 1], association=association,
                    ))
                    for index, (counts, samples) in enumerate(read, start=1)
                ]
            except PreffectError as error:
                st.session_state.pop("prepared", None)
                st.error(str(error))
            else:
                st.session_state["prepared"] = {
                    "written": written,
                    "report": inspect_split(Path(folder), len(read)),
                    "tissues": len(read),
                }

    prepared = st.session_state.get("prepared")
    if prepared is None:
        return
    for path in prepared["written"]:
        st.write(f"wrote `{path}`")
    tissues = f" --tissues {prepared['tissues']}" if prepared["tissues"] > 1 else ""
    st.caption(
        f"The same from a terminal: "
        f"`preffect check --train {Path(prepared['written'][0]).parent}{tissues}`"
    )
    _show_report(st, prepared["report"])


def _fit_page(st, plt, run: Run | None, n_tissues: int) -> None:
    """What was fitted, how the fit went, and what it was applied to."""
    st.header(FIT_PAGE)
    st.markdown(
        "Everything about this fit in one place, before any figure. It says what "
        "kind of model was fitted and on what, how the training loss moved and "
        "whether the run was stopped early, how long it took, and the few "
        "settings that change what the numbers mean. Read it first: a model that "
        "stopped at its first epoch, or one whose validation loss turned upward "
        "and kept going, makes every picture after this one hard to interpret."
    )
    if run is None:
        st.info(
            f"No run is open. Give a folder of counts and a saved model on "
            f"{START_PAGE}, then press Open the run."
        )
        return

    checkpoint, target, result = run.checkpoint, run.target, run.result

    st.subheader("What was fitted")
    row = st.columns(4)
    row[0].metric("model", checkpoint.model_config.kind.value)
    # Which count distribution the model was fitted with. Shown rather than
    # chosen: this page applies a saved model and does not fit one, so the
    # choice belongs to `preffect train --likelihood`. A zero-inflated model
    # and a plain one of the same kind are otherwise indistinguishable here,
    # and the difference changes what the predicted counts mean.
    row[1].metric(
        "count distribution",
        checkpoint.model_config.likelihood.value,
        help="What preffect train --likelihood chose. NB is the negative "
        "binomial. ZINB adds a separate probability that a count is a "
        "structural zero, and predicts different counts.",
    )
    row[2].metric("samples", target.n_obs)
    row[3].metric("genes matched", target.n_vars)

    row = st.columns(4)
    row[0].metric("tissue files", int(checkpoint.extra.get("n_tissues", n_tissues)))
    row[1].metric(
        "library size",
        "inferred" if checkpoint.model_config.infer_library_size else "read",
        help="A sample's library size is its total count. Read means the model "
        "takes the observed total as given; inferred means it fits the total "
        "as well, which it can get wrong.",
    )
    row[2].metric("batch size", _setting(checkpoint.full_config, "train", "batch_size"))
    row[3].metric(
        "learning rate", _setting(checkpoint.full_config, "train", "learning_rate")
    )
    st.caption(f"Written by PREFFECT {checkpoint.written_by}.")
    if result.n_padded_rows_dropped:
        st.caption(
            f"{result.n_padded_rows_dropped} padded row(s) were dropped from the "
            f"result: patients with no file for the tissue the model predicts."
        )

    st.subheader("How the fit went")
    record = checkpoint.training
    if record is None:
        st.info(
            "This model was saved before its history was kept, so there is no "
            "loss curve, no epoch count and no timing to show. Everything else "
            "on this page still applies. A model fitted with this version of "
            "PREFFECT carries its history; refit if you need it."
        )
    else:
        row = st.columns(4)
        row[0].metric(
            "epochs", f"{record.n_epochs} of {record.epochs_asked_for}",
            help="Epochs run, and epochs asked for. They differ after an early stop.",
        )
        row[1].metric("stopped early", "yes" if record.stopped_early else "no")
        row[2].metric(
            "best epoch",
            "not recorded" if record.best_epoch is None else record.best_epoch,
            help="The epoch whose weights were saved. After an early stop that "
            "is the best one rather than the last.",
        )
        row[3].metric("time to fit", _duration(record.wall_seconds))
        st.write(record.stop_reason)
        if record.watched_training_loss:
            st.caption(
                "Early stopping watched the training loss, for want of a "
                "validation split. Stopping on the training loss is a weaker "
                "thing: it says the model stopped improving on what it was "
                "shown, not on anything it had not seen."
            )
        _show(st, plt, lambda: _loss_figure(record))
        st.caption(
            "Mean loss per batch against epoch. A curve still falling steeply "
            "at the last epoch means the fit had not finished; one that is flat "
            "for a long stretch means the later epochs bought nothing. Where a "
            "validation line is drawn, it turning upward while the training "
            "line keeps falling is the model learning the training samples "
            "themselves."
        )

    st.subheader("The cleaned counts")
    st.markdown(
        "The same file `preffect apply` writes, for taking into other tools. "
        "The cleaned counts are in `X`, the predicted proportions in "
        "`layers['composition']`, and the latent summary in `obsm['latent']`."
    )
    adata = result.to_anndata(
        sample_names=[str(n) for n in target.obs_names], gene_names=run.gene_names
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cleaned.h5ad"
        adata.write_h5ad(path)
        payload = path.read_bytes()
    st.download_button(
        "Download the cleaned counts (.h5ad)",
        data=payload,
        file_name="cleaned.h5ad",
    )


def _figure_page(st, plt, plots, page: str, run: Run) -> None:
    """One figure, with what it shows and how to read it."""
    st.header(page)
    st.markdown(FIGURE_DESCRIPTIONS[page])
    choices = _figure_controls(st, page, run, key_prefix="")
    if choices is None:
        return
    _show(st, plt, lambda: _draw_figure(plots, page, run, choices))


def _lookup_page(st, plt, plots, run: Run) -> None:
    """One gene, or one sample, rather than the whole cohort at once."""
    import numpy as np

    from preffect.inference import correlation_per_gene, log_likelihood_per_sample

    st.header(LOOKUP_PAGE)
    st.markdown(
        "The pages before this one describe the whole cohort at once, and this "
        "one answers a question about a single gene or a single sample. Pick a "
        "gene and you see the count the model fitted for it in every sample "
        "against the count observed, how variable the model thinks it is, and "
        "how often it is zero. Pick a sample and you get one number saying how "
        "well the model accounts for it, and where that number sits among the "
        "rest of the cohort. The part worth your time is the list at the "
        "bottom, which is the cohort ordered worst fit first. A sample at the "
        "top of that list is usually a failed library, a mislabelled sample or "
        "a sample of a different tissue from the rest, and it is worth looking "
        "at before it goes into anything downstream."
    )

    result, observed, gene_names = run.result, run.observed, run.gene_names

    st.subheader("One gene")
    gene = st.selectbox("Gene", gene_names, key="gene")
    column = gene_names.index(gene)
    correlations = correlation_per_gene(result, observed)
    dispersion = _as_dense(run.result.dispersion.detach().cpu())[:, column]

    row = st.columns(4)
    row[0].metric("observed mean count", f"{observed[:, column].mean():.1f}")
    row[1].metric("fitted mean count",
                  f"{_as_dense(result.expected_counts.detach().cpu())[:, column].mean():.1f}")
    row[2].metric(
        "correlation across samples",
        "undefined" if np.isnan(correlations[column]) else f"{correlations[column]:.2f}",
        help="Observed against fitted, down the samples. One means the model "
        "tracks this gene's variation perfectly; zero means it predicts the "
        "same count for every sample. Undefined when the gene never moves.",
    )
    row[3].metric("samples where it is zero",
                  f"{float((observed[:, column] == 0).mean()):.0%}")

    _show(st, plt, lambda: plots.gene_scatter(
        result, observed, gene_names=gene_names, genes=[gene]
    ))
    st.caption(
        f"One point per sample: the count observed for {gene} across, the count "
        f"the model fitted up, and the dashed line where the two agree."
    )

    row = st.columns(2)
    row[0].metric("median fitted dispersion", f"{float(np.median(dispersion)):.3g}")
    if result.zero_logits is None:
        row[1].metric(
            "median chance of a zero", "not fitted",
            help="Only the zero-inflated model fits a separate probability of "
            "a zero. This one is a plain negative binomial.",
        )
    else:
        logits = _as_dense(result.zero_logits.detach().cpu())[:, column]
        row[1].metric(
            "median chance of a zero",
            f"{float(np.median(1.0 / (1.0 + np.exp(-logits)))):.2f}",
        )
    _show(st, plt, lambda: plots.gene_dispersion_and_zeros(
        result, observed, gene_names=gene_names, gene=gene
    ))

    st.subheader("One sample")
    st.markdown(
        "The measure is the **mean log probability per gene** of that sample's "
        "observed counts under the distribution the model fitted for it. "
        "Larger is better, and every value is negative, because these are "
        "probabilities of a particular count. It is taken per gene rather than "
        "summed over genes so that it does not grow with the size of the panel. "
        "The number on its own says little; what it is for is the ordering."
    )
    scores = log_likelihood_per_sample(result, observed)
    sample_names = [str(n) for n in run.target.obs_names]
    order = np.argsort(scores)  # worst fitted first
    sample = st.selectbox("Sample", sample_names, key="sample")
    position = sample_names.index(sample)
    rank = int(np.where(order == position)[0][0]) + 1

    row = st.columns(3)
    row[0].metric("mean log probability per gene", f"{float(scores[position]):.3f}")
    row[1].metric("rank, worst fitted first", f"{rank} of {len(sample_names)}")
    row[2].metric("cohort median", f"{float(np.median(scores)):.3f}")
    _show(st, plt, lambda: plots.sample_fit_spread(scores, highlight=position))
    st.caption(
        "Every sample in the cohort, with the one chosen above marked. A long "
        "tail to the left is the handful of samples the model cannot account for."
    )

    st.markdown("**The worst-fitted samples**")
    worst = min(10, len(sample_names))
    st.table(
        [
            {
                "rank": i + 1,
                "sample": sample_names[position],
                "mean log probability per gene": round(float(scores[position]), 3),
            }
            for i, position in enumerate(order[:worst])
        ]
    )


def _impute_page(st, run: Run, device) -> None:
    """Whether this model can fill in a count it was not shown."""
    from preffect.errors import PreffectError

    st.header(IMPUTE_PAGE)
    st.markdown(
        "Imputation is filling in a count that is missing, and this page asks "
        "whether this model can do it for this cohort. It hides a fraction of "
        "the entries at random, runs the model on what is left, and compares "
        "what it predicted at the hidden entries against what was really "
        "there. The number is the average relative distance from the observed "
        "count, so 0.5 means the typical filled-in entry is out by half of the "
        "count that was there, and smaller is better. It comes with how many "
        "entries it was scored over and how many were left out, because a "
        "relative error cannot be computed where the true value is zero. Read "
        "it as a statement about this cohort's own hidden entries and not "
        "about new samples: the model has already seen every sample here, and "
        "filling a gap for a patient it has never met is a harder question "
        "this page does not ask."
    )
    st.markdown(
        "**How to read the number.** On the worked example a simple model "
        "fitted for 80 epochs scores about 0.65 this way. Most of that is not "
        "error: a count is one draw from a negative binomial, and nothing can "
        "predict the draw. Where the file carries the count each entry was "
        "drawn from, as the simulated example data does, scoring against that "
        "instead separates the two, and the same model scores about 0.12. The "
        "worked example page has the exact figures."
    )

    row = st.columns(3)
    fraction = row[0].slider(
        "Fraction of entries to hide", min_value=0.05, max_value=0.5, value=0.2,
        step=0.05, key="hide_fraction",
        help="Taken within every gene, so the same proportion of each gene's "
        "samples is hidden.",
    )
    seed = int(row[1].number_input("Seed", min_value=0, value=0, key="hide_seed"))
    run_it = row[2].button("Hide the entries and score", key="impute")

    if run_it:
        try:
            with st.spinner("Hiding entries and running the model"):
                st.session_state["imputation"] = _score_imputation(
                    run, fraction, seed, device
                )
        except PreffectError as error:
            st.session_state.pop("imputation", None)
            st.error(str(error))

    scored = st.session_state.get("imputation")
    if scored is None:
        st.info("Choose a fraction and a seed, then press the button.")
        return
    if scored["model"] != run.name:
        st.info(
            f"These numbers are for {scored['model']}, which is no longer the "
            f"open run. Press the button again to score {run.name}."
        )
        return

    st.subheader("Against the counts that were observed")
    row = st.columns(3)
    row[0].metric("imputation error", f"{scored['observed']['hidden']:.3f}")
    row[1].metric("entries scored",
                  f"{scored['observed']['n_hidden'] - scored['observed']['skipped']:,}")
    row[2].metric("left out, true value zero", f"{scored['observed']['skipped']:,}")
    st.caption(
        f"{fraction:.0%} of the entries hidden with seed {seed}. The error on "
        f"the entries the model was shown is "
        f"{scored['observed']['visible']:.3f}; doing well on those is not "
        f"imputation, and the two being far apart is the model remembering "
        f"what it saw."
    )

    if scored["truth"] is None:
        st.info(
            "This file carries no layer of the counts each entry was drawn "
            "from, so the comparison above is the honest one available. "
            "PREFFECT will not invent a truth to score against."
        )
        return
    st.subheader("Against the counts they were drawn from")
    st.markdown(
        "This file carries `layers['true_expected_counts']`, the mean each "
        "count was drawn from, which only simulated data can have. That is the "
        "honest truth for imputation: the observed count is one draw from a "
        "distribution, so part of the distance above is noise nothing could "
        "have predicted, and this comparison takes it out."
    )
    row = st.columns(3)
    row[0].metric("imputation error", f"{scored['truth']['hidden']:.3f}")
    row[1].metric("entries scored",
                  f"{scored['truth']['n_hidden'] - scored['truth']['skipped']:,}")
    row[2].metric("left out, true value zero", f"{scored['truth']['skipped']:,}")


def _score_imputation(run: Run, fraction: float, seed: int, device) -> dict:
    """Hide a fraction of the counts, run the model, and score what it filled in.

    The headline number comes from the same function the command line's
    ``--hide`` reports, so the page and the terminal cannot disagree. The
    comparison against the simulated truth needs its own pass, because that
    function scores against the observed counts by definition.
    """
    import numpy as np
    import torch

    from preffect.inference import imputation_error, infer
    from preffect.training import hidden_entry_scores
    from preffect.workflow import hide_entries, to_model_data

    checkpoint = run.checkpoint
    model = checkpoint.build_model()
    model.to(device)
    data = to_model_data(
        run.tissues, checkpoint.model_config, checkpoint.model_config.kind.uses_adjacency
    )
    hidden = hide_entries(data, fraction, seed)
    observed_scores = hidden_entry_scores(model, hidden)

    truth_scores = None
    layer = run.target.layers.get("true_expected_counts")
    if layer is not None:
        result = infer(model, hidden)
        truth = torch.from_numpy(np.asarray(_as_dense(layer), dtype=np.float32))
        truth_scores = imputation_error(
            result.expected_counts.detach().cpu(),
            truth,
            hidden=torch.from_numpy(hidden.mask.hidden),
            kind="mean_relative",
        )

    def tidy(scores: dict) -> dict:
        return {
            "hidden": float(scores["hidden"]),
            "visible": float(scores["visible"]),
            "n_hidden": int(scores["n_hidden"]),
            "skipped": int(scores["zero_truth_entries_skipped"]),
        }

    return {
        "model": run.name,
        "fraction": fraction,
        "seed": seed,
        "observed": tidy(observed_scores),
        "truth": None if truth_scores is None else tidy(truth_scores),
    }


def _compare_page(st, plt, plots, first: Run, second: Run, device) -> None:
    """Two saved models on the same counts, fact by fact and figure by figure."""
    st.header(COMPARE_PAGE)
    st.markdown(
        "Two saved models applied to the same counts, so that the difference "
        "you see is the model and not the data. The first section is what each "
        "run recorded about itself, the second draws any figure under both "
        "models on one set of axes, and the third is a table of the numbers "
        "you would decide on. **It compares two particular fits, not two "
        "methods.** Fitting either model again from a different seed gives a "
        "different answer, which is why this project's own experiments use six "
        "seeds, so treat a small difference here as nothing at all."
    )

    st.subheader("What each run recorded")
    st.table([_facts(first), _facts(second)])

    st.subheader("The same figure under both models")
    page = st.selectbox("Figure", list(FIGURE_PAGES), key="compare_figure")
    st.markdown(FIGURE_DESCRIPTIONS[page])
    choices = _figure_controls(st, page, first, key_prefix="compare_")
    if choices is not None:
        _draw_both(st, plt, plots, page, first, second, choices)

    st.subheader("The numbers you decide on")
    st.markdown(
        "Worked out only when you ask, because clustering the latent space "
        "costs a minute on a real cohort. A row reads **neither** when the two "
        "runs differ by less than a threshold set for this table: 0.01 for a "
        "correlation, 0.05 for the adjusted Rand index and five per cent for "
        "the imputation error. Those thresholds are there to stop a difference "
        "too small to act on being read as a result. They are not a "
        "measurement of the noise, which one fit of each model cannot give."
    )
    columns = _group_columns(first.target.obs)
    label = st.selectbox(
        "Column the clustering is scored against",
        ["none"] + columns,
        key="compare_label",
    ) if columns else "none"
    if st.button("Work out the numbers", key="compare_numbers"):
        with st.spinner("Scoring both runs"):
            st.session_state["comparison"] = _compare_numbers(
                first, second, label, device
            )

    rows = st.session_state.get("comparison")
    if rows is None:
        st.info("Press the button to work the numbers out.")
        return
    st.table(rows)


def _facts(run: Run) -> dict:
    """One run's own account of itself, as a row of a two-row table."""
    record = run.checkpoint.training
    row = {
        "run": run.name,
        "model": run.checkpoint.model_config.kind.value,
        "count distribution": run.checkpoint.model_config.likelihood.value,
    }
    if record is None:
        row |= {
            "epochs": "not recorded",
            "stopped early": "not recorded",
            "final training loss": "not recorded",
            "final validation loss": "not recorded",
            "time to fit": "not recorded",
        }
        return row
    row |= {
        "epochs": f"{record.n_epochs} of {record.epochs_asked_for}",
        "stopped early": "yes" if record.stopped_early else "no",
        "final training loss": f"{record.train_loss[-1]:.2f}",
        "final validation loss": (
            "none" if not record.validation_loss else f"{record.validation_loss[-1]:.2f}"
        ),
        "time to fit": _duration(record.wall_seconds),
    }
    return row


def _draw_both(st, plt, plots, page: str, first: Run, second: Run, choices: dict) -> None:
    """The same figure for two runs, on one set of limits.

    Both are drawn before either is shown, because matching the limits needs
    both to exist. A model that cannot draw this figure says so in its own
    sentence and the other is still shown.
    """
    from preffect.errors import PreffectError

    figures = {}
    for run in (first, second):
        try:
            figures[run.name] = _draw_figure(plots, page, run, choices)
        except PreffectError as error:
            st.info(f"{run.name}: {error}")
    if len(figures) == 2:
        _match_axes(*figures.values())
    for name, figure in figures.items():
        st.markdown(f"**{name}**")
        st.pyplot(figure)
        plt.close(figure)


def _compare_numbers(first: Run, second: Run, label: str, device) -> list[dict]:
    """The table two runs are decided on, one row per measure.

    Every row carries both values and a verdict. A measure that cannot be
    computed is reported as the sentence saying why rather than left out, so
    the table says what was not answered as well as what was.
    """
    import numpy as np

    from preffect.errors import PreffectError
    from preffect.inference import correlation_per_gene

    rows = []

    name = "median correlation, fitted against observed"
    values = {
        run.name: float(np.nanmedian(correlation_per_gene(run.result, run.observed)))
        for run in (first, second)
    }
    rows.append(_row(name, values, larger_is_better=True,
                     threshold=DECIDABLE_DIFFERENCE[name]))

    name = "latent clustering against the chosen column (adjusted Rand)"
    if label == "none":
        rows.append({"measure": name, first.name: "not scored", second.name: "not scored",
                     "better": "no column chosen to score against"})
    else:
        try:
            values = {
                run.name: _clustering_score(run, label) for run in (first, second)
            }
        except PreffectError as error:
            rows.append({"measure": name, first.name: "not scored",
                         second.name: "not scored", "better": str(error)})
        else:
            rows.append(_row(name, values, larger_is_better=True,
                             threshold=DECIDABLE_DIFFERENCE[name]))

    name = "imputation error on hidden entries"
    scored = _imputation_for_both(first, second, device)
    if scored is None:
        rows.append({
            "measure": name, first.name: "not scored", second.name: "not scored",
            "better": f"answer {IMPUTE_PAGE} first; this reuses its fraction and seed",
        })
    else:
        rows.append(_row(name, scored, larger_is_better=False, relative=True,
                         threshold=DECIDABLE_RELATIVE_DIFFERENCE))
    return rows


def _row(measure: str, values: dict, *, larger_is_better: bool,
         threshold: float, relative: bool = False) -> dict:
    """One measure for two runs, with which of them it favours.

    ``relative`` compares the difference against the smaller value rather than
    against a fixed amount, for a measure whose scale depends on the cohort.
    """
    (first_name, first_value), (second_name, second_value) = values.items()
    difference = abs(first_value - second_value)
    scale = min(abs(first_value), abs(second_value)) if relative else 1.0
    undecidable = difference <= threshold * (scale if relative else 1.0)

    if undecidable:
        verdict = "neither, the difference is below what this table can act on"
    else:
        better_first = (first_value > second_value) == larger_is_better
        winner = first_name if better_first else second_name
        if relative and scale > 0:
            verdict = f"{winner}, by {difference / scale:.0%}"
        else:
            verdict = f"{winner}, by {difference:.3f}"
    return {
        "measure": measure,
        first_name: round(first_value, 4),
        second_name: round(second_value, 4),
        "better": verdict,
    }


def _run_label(path: str, taken: str | None) -> str:
    """A short name for a run, lengthened until it is not the other one's.

    The comparison table has one column per run, named after the file, so two
    models both called ``model.pt`` in different folders would collapse into
    one column and the page would quietly show one run twice.
    """
    here = Path(path)
    if taken is None or here.name != taken:
        return here.name
    if str(Path(here.parent.name) / here.name) != taken:
        return str(Path(here.parent.name) / here.name)
    return str(here)


def _clustering_score(run: Run, label: str) -> float:
    """Adjusted Rand index of the latent clustering against a known column."""
    from preffect.clustering import cluster_samples, clustering_recovery

    truth = run.target.obs[label].astype(str).to_numpy()
    groups = max(2, len(set(truth.tolist())))
    found = cluster_samples(run.result, "latent", target_clusters=groups)
    return float(clustering_recovery(found, truth).adjusted_rand)


def _imputation_for_both(first: Run, second: Run, device) -> dict | None:
    """Both runs' imputation error, at the fraction and seed the other page used.

    ``None`` until that page has been run once, because hiding entries and
    running the model is a cost the reader should choose, and because the
    fraction and the seed are theirs to pick rather than this page's to guess.
    """
    import streamlit as st

    scored = st.session_state.get("imputation")
    if scored is None:
        return None
    values = {}
    for run in (first, second):
        if scored["model"] == run.name:
            values[run.name] = scored["observed"]["hidden"]
        else:
            values[run.name] = _score_imputation(
                run, scored["fraction"], scored["seed"], device
            )["observed"]["hidden"]
    return values


def _start_page(st) -> None:
    """Open a run, see what is open, and find out where to go next."""
    st.header(START_PAGE)
    st.markdown(
        "This is for looking at what PREFFECT did: your data before a fit, and "
        "a saved model after one. Fitting itself happens on the command line, "
        "with `preffect train`, because a real cohort takes hours and a "
        "terminal can be left running; what you open here is the model it "
        "wrote, applied to a folder of counts."
    )
    # Filled in after the buttons below have been handled, so that a run
    # opened on this very press is what the line reports.
    state = st.empty()

    st.markdown(
        "**Nothing to open yet?** The button writes the simulated example "
        "cohort, fits the worked example's two models on it, a plain one and a "
        "zero-inflated one, and opens both. It takes a few seconds, and the "
        "paths it fills in below are where it put them."
    )
    if st.button("Try it on the example data", key="example"):
        _try_the_example(st, st.session_state.get("device", "cpu"))

    st.subheader("Your counts")
    row = st.columns([3, 1])
    data_folder = row[0].text_input(
        "Folder of counts", key="data_folder",
        help="One file per tissue, named <anything>tau_<n>.h5ad. The pages "
        "under Your data need only this.",
    )
    n_tissues = int(row[1].number_input(
        "Tissue files in the folder", min_value=1, value=1, key="n_tissues"
    ))

    st.subheader("Your model")
    row = st.columns([3, 1])
    model_path = row[0].text_input(
        "Saved model (.pt)", key="model_path", help="What preffect train wrote."
    )
    device = row[1].selectbox("Run on", ["cpu", "cuda"], key="device")
    if st.button("Open the run", key="apply"):
        if not data_folder or not model_path:
            st.error("Give both a folder of counts and a saved model.")
        elif _apply_model(st, model_path, "run", data_folder, n_tissues, device):
            st.rerun()

    st.subheader("A second model, to compare")
    second_path = st.text_input(
        "Second saved model (.pt)", key="second_model_path",
        help="Optional. Applied to the same counts, which adds the comparison page.",
    )
    if st.button("Open the comparison", key="compare"):
        if not data_folder or not second_path:
            st.error("Give a folder of counts and a second saved model.")
        elif st.session_state.get("run") is None:
            st.error("Open the first run before comparing a second model with it.")
        elif _apply_model(st, second_path, "second_run", data_folder, n_tissues, device):
            st.rerun()

    run, second = st.session_state.get("run"), st.session_state.get("second_run")
    if run is None:
        state.info(
            "No run is open. Give a folder of counts and a saved model below, "
            "then press Open the run. The pages under Your data need only the "
            "folder."
        )
    else:
        beside = f", with `{second.name}` beside it to compare" if second else ""
        state.success(
            f"Open: `{run.name}` on `{data_folder}`, {run.target.n_obs} samples "
            f"and {run.target.n_vars} genes matched{beside}. The pages on the "
            f"left are ready."
        )

    with st.expander("How the pages fit together"):
        st.markdown(
            f"1. **Your data.** *{PREPARE_PAGE}* writes the file PREFFECT reads "
            f"from a counts table and a sample table; *{CHECK_PAGE}* is the pass "
            f"or fail `preffect check` gives; *{LOOK_PAGE}* shows what the folder "
            f"holds, for choosing settings. None of them needs a model.\n"
            f"2. **The fit.** Open a run above and *{FIT_PAGE}* describes the "
            f"saved model: what was fitted, how the loss moved, why it stopped, "
            f"and the cleaned counts to download.\n"
            f"3. **Does the fit hold?** One page per figure, each opening with "
            f"what the picture shows and what a good one looks like, from the "
            f"simplest check to the latent summary.\n"
            f"4. **Ask the model.** One gene or one sample, and whether the model "
            f"can fill in counts it was not shown. **Compare** appears once a "
            f"second model is open on the same counts."
        )


def _try_the_example(st, device) -> None:
    """Write the example cohort, fit the worked example's two models, open both.

    The fit goes through the command line's own ``train``, in this process,
    with the worked example's settings, so what the reader sees is exactly
    what the worked example describes. A refusal arrives as the sentence the
    command line printed. On success the form is filled in on the next run,
    because a field's value cannot be changed after the field has been drawn.
    """
    import contextlib
    import io
    import tempfile

    from preffect.cli import main
    from preffect.errors import PreffectError
    from preffect.example_data import write_example_data

    folder = Path(tempfile.mkdtemp(prefix="preffect_example_"))
    try:
        with st.spinner("Writing the example cohort"):
            write_example_data(folder / "example_data")
    except PreffectError as error:
        st.error(str(error))
        return
    train = folder / "example_data" / "simple" / "train"
    validation = folder / "example_data" / "simple" / "validation"

    models = {}
    for name, likelihood, seed in (("plain.pt", "NB", "0"), ("zero_inflated.pt", "ZINB", "1")):
        out = folder / name
        said = io.StringIO()
        with (
            st.spinner(f"Fitting {name} on the example, with the worked example's settings"),
            contextlib.redirect_stdout(said),
            contextlib.redirect_stderr(said),
        ):
            code = main([
                "train", "--train", str(train), "--validation", str(validation),
                "--out", str(out), "--likelihood", likelihood, "--seed", seed,
                "--device", device, *EXAMPLE_RECIPE,
            ])
        if code != 0:
            lines = [line for line in said.getvalue().splitlines() if line.strip()]
            st.error(lines[-1] if lines else f"fitting {name} failed")
            return
        models[name] = out

    if not _apply_model(st, str(models["plain.pt"]), "run", str(train), 1, device):
        return
    if not _apply_model(st, str(models["zero_inflated.pt"]), "second_run", str(train), 1, device):
        return
    st.session_state["fill_form"] = {
        "data_folder": str(train),
        "n_tissues": 1,
        "model_path": str(models["plain.pt"]),
        "second_model_path": str(models["zero_inflated.pt"]),
    }
    st.rerun()


def _apply_model(st, path: str, slot: str, data_folder: str, n_tissues: int, device) -> bool:
    """Run one saved model over the counts and keep the result; True if it opened.

    Opening the first run clears the second and everything worked out from
    the pair of them. Otherwise a reader who points the interface at a
    different folder and reopens is comparing two models over two different
    sets of counts, with nothing on the page saying so.
    """
    from preffect.config import RuntimeConfig
    from preffect.errors import PreffectError
    from preffect.training import resolve_device
    from preffect.workflow import apply_checkpoint

    try:
        with st.spinner("Applying the model"):
            resolved = resolve_device(RuntimeConfig(device=device))
            applied = apply_checkpoint(
                Path(path), Path(data_folder), n_tissues=n_tissues, device=resolved
            )
    except PreffectError as error:
        st.error(str(error))
        return False

    if slot == "run":
        for stale in ("second_run", "comparison", "imputation"):
            st.session_state.pop(stale, None)
        st.session_state["run"] = Run(_run_label(path, None), *applied)
        return True
    st.session_state.pop("comparison", None)
    taken = st.session_state["run"].name
    st.session_state[slot] = Run(_run_label(path, taken), *applied)
    return True


def _status_block(st, run: Run | None, second: Run | None, data_folder: str) -> None:
    """Three lines under the page list saying what is open, on every page."""
    st.markdown("**What is open**")
    if run is not None:
        st.caption(
            f"Counts: `{data_folder}`, {run.target.n_obs} samples, "
            f"{run.target.n_vars} genes matched, {len(run.tissues)} tissue file(s)."
        )
    elif data_folder:
        st.caption(f"Counts: `{data_folder}`, not opened yet.")
    else:
        st.caption("Counts: no folder open yet.")
    if run is None:
        st.caption("Model: no model open yet.")
    else:
        config = run.checkpoint.model_config
        st.caption(
            f"Model: `{run.name}`, the {config.kind.value} model, "
            f"{config.likelihood.value} distribution."
        )
    if second is None:
        st.caption("Second model: not open.")
    else:
        st.caption(f"Second model: `{second.name}`, open for comparison.")


def _available(page: str, run: Run | None, second: Run | None) -> bool:
    """Whether a page has what it needs, so that it is listed only then."""
    if page == COMPARE_PAGE:
        return second is not None
    if page == START_PAGE or page in DATA_PAGES:
        return True
    return run is not None


def render() -> None:
    """Build the page list from what is open, and draw the chosen page.

    Streamlit runs this again on every interaction.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import streamlit as st

    from preffect import __version__, plots

    st.set_page_config(page_title="PREFFECT", layout="wide")
    # Streamlit forgets a widget's value as soon as a run passes without
    # drawing it, which is every visit to another page. Writing each value
    # back on every run is the documented way to keep it.
    for key in FORM_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    # What the example button asked to be filled in, applied here because a
    # field's value can only be set before the field is drawn.
    st.session_state.update(st.session_state.pop("fill_form", {}))

    st.title(f"PREFFECT {__version__}")
    st.caption("Runs on this machine only. Nothing you point it at leaves it.")

    run = st.session_state.get("run")
    second = st.session_state.get("second_run")
    data_folder = st.session_state.get("data_folder", "")
    n_tissues = int(st.session_state.get("n_tissues", 1))
    device = st.session_state.get("device", "cpu")

    drawers = {
        START_PAGE: lambda: _start_page(st),
        PREPARE_PAGE: lambda: _prepare_page(st),
        CHECK_PAGE: lambda: _check_page(st, data_folder, n_tissues),
        LOOK_PAGE: lambda: _look_page(st, plt, plots, data_folder, n_tissues),
        FIT_PAGE: lambda: _fit_page(st, plt, run, n_tissues),
        LOOKUP_PAGE: lambda: _lookup_page(st, plt, plots, run),
        IMPUTE_PAGE: lambda: _impute_page(st, run, device),
        COMPARE_PAGE: lambda: _compare_page(st, plt, plots, run, second, device),
    }
    for title in FIGURE_PAGES:
        drawers[title] = lambda title=title: _figure_page(st, plt, plots, title, run)

    # The sidebar lists a page only when what it needs is open, so a page is
    # never offered with nothing behind it. Only the chosen page is drawn.
    sections = {}
    for heading, titles in SECTIONS:
        offered = [t for t in titles if _available(t, run, second)]
        if offered:
            sections[heading] = [
                st.Page(drawers[t], title=t, url_path=URL_PATHS[t], default=t == START_PAGE)
                for t in offered
            ]
    st.navigation(sections, position="sidebar").run()

    # Drawn after the page, so that a run opened on it is what the lines say.
    with st.sidebar:
        _status_block(
            st, st.session_state.get("run"), st.session_state.get("second_run"),
            st.session_state.get("data_folder", ""),
        )


if __name__ == "__main__":
    render()
