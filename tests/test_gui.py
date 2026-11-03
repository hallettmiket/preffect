"""The graphical interface, driven without a browser.

Streamlit's test harness runs the page as a script and exposes what it drew.
These tests fill the sidebar in, move between pages, and check three things:
that every page draws without an exception, that each one opens by saying what
it is for, and that a problem shows as a sentence rather than a traceback.

The interface is a report with one page per figure, so the thing most worth
holding is that a page is never offered with nothing behind it and never
arrives without its description. A figure with no words around it is the
failure this structure exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests.conftest import make_counts, write_split

import preffect.gui
from preffect.checkpoint import TRAINING_HISTORY_KEY, load_checkpoint
from preffect.cli import main
from preffect.gui import (
    COMPARE_PAGE,
    FIGURE_DESCRIPTIONS,
    FIGURE_PAGES,
    IMPUTE_PAGE,
    LOOKUP_PAGE,
    OPENING_PAGES,
    TOOL_PAGES,
)

pytestmark = pytest.mark.extras

PAGE = Path(preffect.gui.__file__)

#: The comparison table's clustering row, named once so a test asking for it
#: and the table writing it cannot drift apart.
_CLUSTERING_MEASURE = "latent clustering against the chosen column (adjusted Rand)"


@pytest.fixture
def trained(tmp_path):
    folder = write_split(tmp_path / "train", [make_counts(n_samples=24, n_genes=12, seed=0)])
    model = tmp_path / "m.pt"
    assert main(["train", "--train", str(folder), "--out", str(model),
                 "--epochs", "2", "--batch-size", "8", "--seed", "0", "--threads", "2"]) == 0
    return folder, model


@pytest.fixture
def trained_without_history(trained, tmp_path):
    """The same model, saved the way files were written before the run's own
    record travelled with them. Built by stripping the record out of a real
    file rather than by hand, so what is tested is a file this package can
    actually be given."""
    folder, model = trained
    payload = torch.load(model, map_location="cpu", weights_only=True)
    payload["extra"] = {
        key: value for key, value in payload["extra"].items() if key != TRAINING_HISTORY_KEY
    }
    older = tmp_path / "older.pt"
    torch.save(payload, older)
    return folder, older


@pytest.fixture
def with_truth_layer(tmp_path):
    """Counts that carry the mean each one was drawn from, as simulated data does.

    Only simulation knows it, and it is the honest truth to score imputation
    against: an observed count is one draw, so part of the distance from it is
    noise nothing could have predicted.
    """
    counts = make_counts(n_samples=24, n_genes=12, seed=0)
    counts.layers["true_expected_counts"] = counts.X.astype("float32") + 1.0
    folder = write_split(tmp_path / "simulated", [counts])
    model = tmp_path / "sim.pt"
    assert main(["train", "--train", str(folder), "--out", str(model),
                 "--epochs", "2", "--batch-size", "8", "--seed", "0", "--threads", "2"]) == 0
    return folder, model


@pytest.fixture
def two_models(tmp_path):
    """Two models fitted on the same counts from different seeds."""
    folder = write_split(tmp_path / "train", [make_counts(n_samples=24, n_genes=12, seed=0)])
    paths = []
    for name, seed in (("first.pt", "0"), ("second.pt", "1")):
        path = tmp_path / name
        assert main(["train", "--train", str(folder), "--out", str(path),
                     "--epochs", "2", "--batch-size", "8", "--seed", seed,
                     "--threads", "2"]) == 0
        paths.append(path)
    return folder, paths[0], paths[1]


def _page():
    testing = pytest.importorskip("streamlit.testing.v1", reason="needs the gui extra")
    return testing.AppTest.from_file(str(PAGE), default_timeout=300)


def _opened(trained):
    """A page with the run open, which is where every figure page starts."""
    folder, model = trained
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(folder))
    at.text_input(key="model_path").set_value(str(model))
    at.button(key="apply").click().run()
    assert not at.exception, at.exception
    return at


def _compared(two_models):
    """A page with both runs open, which is where the comparison starts."""
    folder, first, second = two_models
    at = _opened((folder, first))
    at.text_input(key="second_model_path").set_value(str(second))
    at.button(key="compare").click().run()
    assert not at.exception, at.exception
    return at


def _measure_row(table, measure: str) -> dict:
    """One row of the comparison table, by the measure it reports."""
    match = table[table["measure"] == measure]
    assert len(match) == 1, f"{measure} is not in the table"
    return match.iloc[0].to_dict()


def test_the_page_opens_with_nothing_filled_in():
    at = _page().run()
    assert not at.exception
    assert "PREFFECT" in at.title[0].value


def test_before_a_run_is_open_no_figure_page_is_offered():
    """A page offered with nothing behind it is a page that errors when you
    press it. The figure pages appear once a model has been applied."""
    at = _page().run()
    assert at.radio(key="page").options == list(OPENING_PAGES)
    assert any("No run is open" in m.value for m in at.info)


# --------------------------------------------------------------------------
# Checking a folder, which needs no model.
# --------------------------------------------------------------------------


def test_checking_a_folder_shows_what_is_in_it(trained):
    folder, _ = trained
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(folder))
    at.radio(key="page").set_value("Check the data").run()
    at.button(key="check").click().run()
    assert not at.exception
    assert any("usable" in s.value for s in at.success)
    table = at.table[0].value
    assert int(table["samples"].iloc[0]) == 24
    assert int(table["genes"].iloc[0]) == 12


def test_checking_a_missing_folder_shows_the_sentence_not_a_traceback(tmp_path):
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(tmp_path / "absent"))
    at.radio(key="page").set_value("Check the data").run()
    at.button(key="check").click().run()
    assert not at.exception
    assert any("no such folder" in e.value for e in at.error)


def test_opening_a_run_without_a_model_says_so(trained):
    folder, _ = trained
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(folder))
    at.button(key="apply").click().run()
    assert not at.exception
    assert any("saved model" in e.value for e in at.error)


# --------------------------------------------------------------------------
# The run page: what was fitted, and how the fit went.
# --------------------------------------------------------------------------


def test_the_run_page_says_what_was_fitted(trained):
    at = _opened(trained)
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics["samples"] == "24"
    assert metrics["genes matched"] == "12"
    assert metrics["count distribution"] == "NB"
    assert metrics["tissue files"] == "1"
    assert metrics["batch size"] == "8"


def test_the_run_page_says_how_many_epochs_it_ran_and_why_it_stopped(trained):
    """The two questions asked of any fit. A model saved by this version
    carries the answer, so the page states it rather than leaving a reader to
    find the terminal the run was started from."""
    at = _opened(trained)
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics["epochs"] == "2 of 2"
    assert metrics["stopped early"] == "no"
    assert any("ran all 2 epoch(s) asked for" in m.value for m in at.markdown)
    assert "time to fit" in metrics


def test_the_run_page_says_which_distribution_the_model_was_fitted_with(tmp_path):
    """The page applies a saved model and does not fit one, so the choice is
    `preffect train --likelihood`. What it owes the reader is which choice was
    made: a zero-inflated model and a plain one of the same kind look the same
    here, and they predict different counts."""
    folder = write_split(tmp_path / "train", [make_counts(n_samples=24, n_genes=12, seed=0)])
    model = tmp_path / "zinb.pt"
    assert main(["train", "--train", str(folder), "--out", str(model),
                 "--likelihood", "ZINB", "--epochs", "2", "--batch-size", "8",
                 "--seed", "0", "--threads", "2"]) == 0

    at = _opened((folder, model))
    assert {m.label: str(m.value) for m in at.metric}["count distribution"] == "ZINB"


def test_the_cleaned_counts_can_still_be_downloaded(trained):
    at = _opened(trained)
    assert len(at.download_button) == 1
    assert "h5ad" in at.download_button[0].label


def test_a_model_saved_before_its_history_was_kept_still_opens(trained_without_history):
    """`extra` is free-form, so a file written before the run's record was put
    there is still a valid checkpoint and still loads. The page says what is
    missing in a sentence, and everything that does not need the history is
    shown as usual."""
    folder, older = trained_without_history
    assert load_checkpoint(older).training is None

    at = _opened((folder, older))
    assert any("saved before its history was kept" in m.value for m in at.info)
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics["samples"] == "24"
    assert "epochs" not in metrics


# --------------------------------------------------------------------------
# One page per figure, each opening with what the picture is for.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("page", FIGURE_PAGES)
def test_every_figure_page_draws_and_opens_with_its_description(trained, page):
    """Each figure page is headed by its own name and opens with a paragraph
    saying what is on the axes, what a good picture looks like and what to do
    about a bad one. A figure with no words around it is what this structure
    exists to prevent, so the words are tested."""
    at = _opened(trained)
    at.radio(key="page").set_value(page).run()
    assert not at.exception, at.exception
    assert at.header[0].value == page
    description = at.markdown[0].value
    assert len(description) > 300
    assert description.count(". ") >= 2


def test_the_figure_pages_appear_only_once_a_run_is_open(trained):
    at = _opened(trained)
    assert at.radio(key="page").options == (
        list(OPENING_PAGES) + list(FIGURE_PAGES) + list(TOOL_PAGES)
    )
    assert COMPARE_PAGE not in at.radio(key="page").options


def test_the_batch_page_compares_the_column_you_name(trained):
    """The sample table decides what a batch is, so the column is chosen on the
    page. The test data carries one called `batch`, which is picked first."""
    at = _opened(trained)
    at.radio(key="page").set_value("Batch adjustment").run()
    assert not at.exception
    assert at.selectbox(key="batch_by").value == "batch"


def test_the_zero_page_says_why_it_cannot_draw_for_a_plain_model(trained):
    """Only the zero-inflated model has a probability of a zero to plot. The
    page is still there, with its description, and says in a sentence what to
    refit with. Version 0.1's habit was to quietly draw something instead."""
    at = _opened(trained)
    at.radio(key="page").set_value("Zeros").run()
    assert not at.exception
    assert any("ZINB" in m.value for m in at.info)


# --------------------------------------------------------------------------
# Gene and sample lookup: one gene, or one sample, rather than the cohort.
# --------------------------------------------------------------------------


def test_the_lookup_page_answers_for_one_gene(trained):
    at = _opened(trained)
    at.radio(key="page").set_value(LOOKUP_PAGE).run()
    assert not at.exception, at.exception
    assert at.header[0].value == LOOKUP_PAGE
    assert len(at.markdown[0].value) > 300

    assert at.selectbox(key="gene").value == "Gene_0"
    metrics = {m.label: str(m.value) for m in at.metric}
    assert "observed mean count" in metrics
    assert "correlation across samples" in metrics
    assert metrics["samples where it is zero"].endswith("%")


def test_the_lookup_page_says_which_measure_ranks_the_samples(trained):
    """One number per sample, the same for every sample, and the page names
    it. A ranking whose measure is not stated cannot be argued with."""
    at = _opened(trained)
    at.radio(key="page").set_value(LOOKUP_PAGE).run()
    said = " ".join(m.value for m in at.markdown)
    assert "mean log probability per gene" in said
    assert "Larger is better" in said
    metrics = {m.label: str(m.value) for m in at.metric}
    assert "rank, worst fitted first" in metrics
    assert metrics["rank, worst fitted first"].endswith("of 24")


def test_the_lookup_page_lists_the_worst_fitted_samples_first(trained):
    at = _opened(trained)
    at.radio(key="page").set_value(LOOKUP_PAGE).run()
    table = at.table[0].value
    scores = [float(v) for v in table["mean log probability per gene"]]
    assert scores == sorted(scores), "the list is not worst fitted first"
    assert list(table["rank"]) == list(range(1, len(scores) + 1))


def test_a_plain_model_says_it_fits_no_chance_of_a_zero(trained):
    """The zero probability exists only for the zero-inflated model, and the
    page says so rather than leaving the reader to wonder."""
    at = _opened(trained)
    at.radio(key="page").set_value(LOOKUP_PAGE).run()
    assert {m.label: str(m.value) for m in at.metric}["median chance of a zero"] == (
        "not fitted"
    )


# --------------------------------------------------------------------------
# Can this model impute?
# --------------------------------------------------------------------------


def test_the_imputation_page_scores_what_the_model_filled_in(trained):
    at = _opened(trained)
    at.radio(key="page").set_value(IMPUTE_PAGE).run()
    assert not at.exception, at.exception
    assert at.header[0].value == IMPUTE_PAGE
    assert len(at.markdown[0].value) > 300
    assert any("press the button" in i.value.lower() for i in at.info)

    at.button(key="impute").click().run()
    assert not at.exception, at.exception
    metrics = {m.label: str(m.value) for m in at.metric}
    assert float(metrics["imputation error"]) > 0
    assert int(metrics["entries scored"].replace(",", "")) > 0
    assert "left out, true value zero" in metrics


def test_the_imputation_page_says_there_is_no_truth_to_score_against(trained):
    """Ordinary counts carry no record of what each entry should have been,
    and PREFFECT will not invent one. Finding 10."""
    at = _opened(trained)
    at.radio(key="page").set_value(IMPUTE_PAGE).run()
    at.button(key="impute").click().run()
    assert any("will not invent a truth" in i.value for i in at.info)


def test_the_imputation_page_scores_against_the_simulated_truth_when_it_is_there(
    with_truth_layer,
):
    """Simulated data carries the mean each count was drawn from, which is the
    honest comparison: the observed count is one draw from a distribution, and
    part of the distance from it is noise nothing could have predicted."""
    at = _opened(with_truth_layer)
    at.radio(key="page").set_value(IMPUTE_PAGE).run()
    at.button(key="impute").click().run()
    assert not at.exception, at.exception
    assert not any("will not invent a truth" in i.value for i in at.info)
    headings = [h.value for h in at.subheader]
    assert "Against the counts that were observed" in headings
    assert "Against the counts they were drawn from" in headings
    assert len([m for m in at.metric if m.label == "imputation error"]) == 2


# --------------------------------------------------------------------------
# Compare two runs.
# --------------------------------------------------------------------------


def test_the_compare_page_appears_once_a_second_model_is_open(two_models):
    at = _compared(two_models)
    assert COMPARE_PAGE in at.radio(key="page").options


def test_comparing_needs_a_first_run_to_compare_against(two_models):
    folder, _, second = two_models
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(folder))
    at.text_input(key="second_model_path").set_value(str(second))
    at.button(key="compare").click().run()
    assert not at.exception
    assert any("Open the first run" in e.value for e in at.error)


def test_the_compare_page_puts_the_two_runs_side_by_side(two_models):
    at = _compared(two_models)
    at.radio(key="page").set_value(COMPARE_PAGE).run()
    assert not at.exception, at.exception
    assert at.header[0].value == COMPARE_PAGE
    assert len(at.markdown[0].value) > 300

    facts = at.table[0].value
    assert list(facts["run"]) == ["first.pt", "second.pt"]
    assert list(facts["epochs"]) == ["2 of 2", "2 of 2"]
    assert set(facts["model"]) == {"simple"}


def test_the_compare_page_draws_one_figure_under_both_models(two_models):
    """The same figure twice, on one set of limits. Two pictures on different
    scales look like two different results even when they are the same one."""
    at = _compared(two_models)
    at.radio(key="page").set_value(COMPARE_PAGE).run()
    at.selectbox(key="compare_figure").set_value("Library sizes").run()
    assert not at.exception, at.exception
    shown = [m.value for m in at.markdown]
    assert "**first.pt**" in shown
    assert "**second.pt**" in shown
    assert any(FIGURE_DESCRIPTIONS["Library sizes"] == m.value for m in at.markdown)


def test_the_compare_page_says_which_run_is_better_on_what(two_models):
    at = _compared(two_models)
    at.radio(key="page").set_value(COMPARE_PAGE).run()
    at.button(key="compare_numbers").click().run()
    assert not at.exception, at.exception

    numbers = at.table[1].value
    measures = list(numbers["measure"])
    assert "median correlation, fitted against observed" in measures
    assert any("adjusted Rand" in m for m in measures)
    assert "imputation error on hidden entries" in measures
    verdicts = " ".join(numbers["better"])
    assert "first.pt" in verdicts or "second.pt" in verdicts or "neither" in verdicts


def test_the_comparison_table_reuses_the_imputation_page_s_own_settings(two_models):
    """Hiding entries and running the model is a cost, and the fraction and
    seed are the reader's to pick, so the row is filled in only after that
    page has been run once."""
    at = _compared(two_models)
    at.radio(key="page").set_value(COMPARE_PAGE).run()
    at.button(key="compare_numbers").click().run()
    row = _measure_row(at.table[1].value, "imputation error on hidden entries")
    assert IMPUTE_PAGE in row["better"]

    at.radio(key="page").set_value(IMPUTE_PAGE).run()
    at.button(key="impute").click().run()
    at.radio(key="page").set_value(COMPARE_PAGE).run()
    at.button(key="compare_numbers").click().run()
    assert not at.exception, at.exception
    row = _measure_row(at.table[1].value, "imputation error on hidden entries")
    assert float(row["first.pt"]) > 0
    assert float(row["second.pt"]) > 0


def test_a_difference_too_small_to_act_on_reads_as_neither():
    """One fit of each model cannot say how much of a difference is the seed,
    so the table refuses to name a winner below a stated threshold rather than
    reporting the sign of the noise."""
    from preffect.gui import _row

    row = _row("a measure", {"a.pt": 0.900, "b.pt": 0.905},
               larger_is_better=True, threshold=0.01)
    assert row["better"].startswith("neither")

    row = _row("a measure", {"a.pt": 0.900, "b.pt": 0.950},
               larger_is_better=True, threshold=0.01)
    assert row["better"] == "b.pt, by 0.050"

    row = _row("an error", {"a.pt": 1.0, "b.pt": 2.0},
               larger_is_better=False, threshold=0.05, relative=True)
    assert row["better"] == "a.pt, by 100%"


def test_the_comparison_scores_the_latent_clustering_against_a_chosen_column(two_models):
    """The clustering row needs a column of known labels to score against, and
    the reader picks it, because which column the groups ought to follow is
    not something the interface can know."""
    at = _compared(two_models)
    at.radio(key="page").set_value(COMPARE_PAGE).run()
    assert at.selectbox(key="compare_label").options == ["none", "batch"]

    at.button(key="compare_numbers").click().run()
    row = _measure_row(at.table[1].value, _CLUSTERING_MEASURE)
    assert row["better"] == "no column chosen to score against"

    at.selectbox(key="compare_label").set_value("batch").run()
    at.button(key="compare_numbers").click().run()
    assert not at.exception, at.exception
    row = _measure_row(at.table[1].value, _CLUSTERING_MEASURE)
    assert -1.0 <= float(row["first.pt"]) <= 1.0
    assert -1.0 <= float(row["second.pt"]) <= 1.0


def test_reopening_the_first_run_drops_the_comparison(two_models, tmp_path):
    """A reader who points the interface at another folder and reopens would
    otherwise be comparing two models over two different sets of counts, with
    nothing on the page saying so."""
    at = _compared(two_models)
    assert COMPARE_PAGE in at.radio(key="page").options

    other = write_split(tmp_path / "other", [make_counts(n_samples=24, n_genes=12, seed=5)])
    at.radio(key="page").set_value("The run").run()
    at.text_input(key="data_folder").set_value(str(other))
    at.button(key="apply").click().run()
    assert not at.exception, at.exception
    assert COMPARE_PAGE not in at.radio(key="page").options


def test_two_models_with_the_same_file_name_get_told_apart(tmp_path):
    """One column per run in the comparison table, named after the file. Two
    models both called model.pt would collapse into one column and the page
    would show one run twice."""
    from preffect.gui import _run_label

    folder = write_split(tmp_path / "train", [make_counts(n_samples=24, n_genes=12, seed=0)])
    paths = []
    for run in ("a", "b"):
        path = tmp_path / run / "model.pt"
        assert main(["train", "--train", str(folder), "--out", str(path),
                     "--epochs", "2", "--batch-size", "8", "--seed", "0",
                     "--threads", "2"]) == 0
        paths.append(path)
    assert _run_label(str(paths[0]), None) == "model.pt"
    assert _run_label(str(paths[1]), "model.pt") == "b/model.pt"

    at = _compared((folder, paths[0], paths[1]))
    at.radio(key="page").set_value(COMPARE_PAGE).run()
    assert not at.exception, at.exception
    assert list(at.table[0].value["run"]) == ["model.pt", "b/model.pt"]


# --------------------------------------------------------------------------
# Prepare data: two tables in, the file the rest of the package reads out.
#
# This is the page a reader meets before they have anything PREFFECT can
# open, so the thing worth holding is that every refusal arrives as its own
# sentence and that the page finishes on what `preffect check` would say.
# --------------------------------------------------------------------------


def _csv(rows) -> bytes:
    return "\n".join(",".join(str(v) for v in row) for row in rows).encode()


def _counts_csv(n_samples: int = 6, n_genes: int = 4, prefix: str = "s") -> bytes:
    """A tiny counts table, samples down the rows."""
    header = ["sample", *[f"Gene_{j}" for j in range(n_genes)]]
    rows = [header] + [
        [f"{prefix}{i}", *[(i * 7 + j * 3) % 29 for j in range(n_genes)]]
        for i in range(n_samples)
    ]
    return _csv(rows)


def _samples_csv(n_samples: int = 6, prefix: str = "s") -> bytes:
    rows = [["sample", "batch", "patient"]] + [
        [f"{prefix}{i}", i % 2, f"p{i}"] for i in range(n_samples)
    ]
    return _csv(rows)


def _uploaded(at, counts: bytes, samples: bytes, tissue: int = 1):
    """Put one tissue's two tables into the page as a browser upload would."""
    at.file_uploader(key=f"prepare_counts_{tissue}").set_value(
        ("counts.csv", counts, "text/csv")
    )
    at.file_uploader(key=f"prepare_samples_{tissue}").set_value(
        ("samples.csv", samples, "text/csv")
    )
    return at


def _on_prepare():
    at = _page().run()
    at.radio(key="page").set_value("Prepare data").run()
    return at


def test_the_prepare_page_waits_for_both_tables():
    at = _on_prepare()
    assert not at.exception
    assert at.header[0].value == "Prepare data"
    assert len(at.markdown[0].value) > 300
    assert any("counts table and a sample table" in i.value for i in at.info)


def test_the_prepare_page_says_what_it_read_and_which_way_round(two_models=None):
    at = _on_prepare()
    _uploaded(at, _counts_csv(), _samples_csv())
    at.run()
    assert not at.exception, at.exception
    read = at.table[0].value
    assert int(read["samples"].iloc[0]) == 6
    assert int(read["genes"].iloc[0]) == 4
    assert int(read["annotations per sample"].iloc[0]) == 2
    said = " ".join(c.value for c in at.caption)
    assert "Read as samples-by-genes" in said
    assert "batch, patient" in said


def test_a_table_written_the_other_way_round_is_flipped_on_request():
    """Many pipelines write one row per gene. Reading it as it stands would
    make four samples of four genes and twenty-nine of six, without a word."""
    at = _on_prepare()
    at.radio(key="prepare_orientation").set_value("genes-by-samples").run()
    _uploaded(at, _counts_csv(n_samples=4, n_genes=6), _samples_csv(n_samples=6))
    at.run()
    assert not at.exception, at.exception
    read = at.table[0].value
    assert int(read["samples"].iloc[0]) == 6
    assert int(read["genes"].iloc[0]) == 4


def test_a_counts_table_that_cannot_be_read_shows_its_sentence(tmp_path):
    """A path that is not there used to reach the page as twenty lines of
    pandas stack, here and on the command line. Finding 66."""
    at = _on_prepare()
    at.radio(key="prepare_source").set_value("Already on this machine").run()
    at.text_input(key="prepare_counts_path_1").set_value(str(tmp_path / "absent.csv"))
    at.text_input(key="prepare_samples_path_1").set_value(str(tmp_path / "absent.csv"))
    at.run()
    assert not at.exception, at.exception
    assert any("could not open the counts table" in e.value for e in at.error)


def test_the_network_summary_moves_with_the_neighbour_count():
    """The number of neighbours is the one choice here with no right answer,
    so what it does to the links per sample is shown as the slider moves."""
    at = _on_prepare()
    _uploaded(at, _counts_csv(n_samples=10, n_genes=5), _samples_csv(n_samples=10))
    at.run()
    at.checkbox(key="prepare_network").set_value(True).run()
    assert not at.exception, at.exception

    at.slider(key="prepare_neighbours").set_value(2).run()
    few = [m.value for m in at.markdown if m.value.startswith("**Tissue 1.**")][0]
    at.slider(key="prepare_neighbours").set_value(5).run()
    many = [m.value for m in at.markdown if m.value.startswith("**Tissue 1.**")][0]
    assert "edge(s)" in few and "edge(s)" in many
    assert few != many


def test_the_network_says_how_often_a_link_joins_the_same_label():
    """Against the share that would join by chance, because the gap between
    the two is what says the network carries biology rather than noise."""
    at = _on_prepare()
    _uploaded(at, _counts_csv(n_samples=10, n_genes=5), _samples_csv(n_samples=10))
    at.run()
    at.checkbox(key="prepare_network").set_value(True).run()
    at.slider(key="prepare_neighbours").set_value(3).run()
    at.selectbox(key="prepare_label").set_value("batch").run()
    assert not at.exception, at.exception
    said = [m.value for m in at.markdown if m.value.startswith("**Tissue 1.**")][0]
    assert "the same batch" in said
    assert "if the edges ignored it" in said


def test_the_neighbour_slider_stops_short_of_linking_everything():
    """The builder refuses a count that links every sample to every other,
    because a complete network says nothing about which samples resemble
    which. The slider does not offer one, so the smallest tissue sets how far
    it goes."""
    at = _on_prepare()
    _uploaded(at, _counts_csv(n_samples=6, n_genes=4), _samples_csv(n_samples=6))
    at.run()
    at.checkbox(key="prepare_network").set_value(True).run()
    assert not at.exception, at.exception
    assert at.slider(key="prepare_neighbours").proto.max == 4.0
    assert at.slider(key="prepare_neighbours").value == 4


def test_keeping_only_the_links_both_samples_name_leaves_some_with_none():
    """A sparser network, and the summary says how many samples it left with
    nothing to attend to, which is what a graph model learns nothing for."""
    at = _on_prepare()
    _uploaded(at, _counts_csv(n_samples=6, n_genes=4), _samples_csv(n_samples=6))
    at.run()
    at.checkbox(key="prepare_network").set_value(True).run()
    at.slider(key="prepare_neighbours").set_value(1).run()
    at.checkbox(key="prepare_mutual").set_value(True).run()
    assert not at.exception, at.exception
    said = [m.value for m in at.markdown if m.value.startswith("**Tissue 1.**")][0]
    assert "with no link at all" in said


def test_two_tissues_ask_which_column_names_the_patient():
    """The files share no sample names, so that column is the only thing
    saying a sample here and a sample there came from one patient."""
    at = _on_prepare()
    at.number_input(key="prepare_tissues").set_value(2).run()
    _uploaded(at, _counts_csv(prefix="a"), _samples_csv(prefix="a"), tissue=1)
    _uploaded(at, _counts_csv(n_samples=4, prefix="b"), _samples_csv(4, prefix="b"),
              tissue=2)
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="prepare_patient").options == ["batch", "patient"]

    at.selectbox(key="prepare_patient").set_value("patient").run()
    assert not at.exception, at.exception
    missing = at.table[1].value
    assert list(missing["patients with a sample here"]) == [6, 4]
    assert list(missing["patients with none"]) == [0, 2]


def test_writing_the_files_ends_on_what_check_says(tmp_path):
    at = _on_prepare()
    _uploaded(at, _counts_csv(), _samples_csv())
    at.run()
    at.text_input(key="prepare_out").set_value(str(tmp_path / "prepared"))
    at.button(key="prepare_write").click().run()
    assert not at.exception, at.exception

    written = (tmp_path / "prepared" / "counts.tau_1.h5ad")
    assert written.is_file()
    assert any(str(written) in m.value for m in at.markdown)
    assert any("usable" in s.value for s in at.success)
    table = at.table[-1].value
    assert int(table["samples"].iloc[0]) == 6
    assert int(table["genes"].iloc[0]) == 4


def test_the_written_file_is_one_preffect_can_actually_fit(tmp_path):
    """The point of the page. Checking is not enough on its own: `check` and
    `train` read the file through the same loader, but only training says the
    annotations and the counts came out usable together."""
    at = _on_prepare()
    _uploaded(at, _counts_csv(n_samples=12, n_genes=6), _samples_csv(n_samples=12))
    at.run()
    folder = tmp_path / "prepared"
    at.text_input(key="prepare_out").set_value(str(folder)).run()
    at.button(key="prepare_write").click().run()
    assert not at.exception, at.exception

    assert main(["train", "--train", str(folder), "--out", str(tmp_path / "m.pt"),
                 "--epochs", "2", "--batch-size", "4", "--seed", "0",
                 "--threads", "2"]) == 0


def test_writing_over_a_file_already_there_is_refused_by_name(tmp_path):
    """Preparing twice into one folder would otherwise lose whatever the
    first run put there, and nothing on the page would say so."""
    at = _on_prepare()
    _uploaded(at, _counts_csv(), _samples_csv())
    at.run()
    at.text_input(key="prepare_out").set_value(str(tmp_path / "prepared")).run()
    at.button(key="prepare_write").click().run()
    assert any("usable" in s.value for s in at.success)

    at.button(key="prepare_write").click().run()
    assert not at.exception, at.exception
    assert any("is there already" in e.value for e in at.error)
    assert not at.success


def test_counts_that_are_not_counts_are_refused_before_anything_is_written(tmp_path):
    """Every entry is checked first, and the message names the ones at fault,
    so a table with one bad cell does not take a run to find out."""
    rows = [["sample", "Gene_0", "Gene_1"]] + [
        [f"s{i}", 3 if i else -1, 4] for i in range(6)
    ]
    at = _on_prepare()
    _uploaded(at, _csv(rows), _samples_csv())
    at.run()
    at.text_input(key="prepare_out").set_value(str(tmp_path / "prepared")).run()
    at.button(key="prepare_write").click().run()
    assert not at.exception, at.exception
    assert any("negative" in e.value.lower() for e in at.error)
    assert not (tmp_path / "prepared" / "counts.tau_1.h5ad").exists()
