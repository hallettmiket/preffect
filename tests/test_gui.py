"""The graphical interface, driven without a browser.

Streamlit's test harness runs the page as a script and exposes what it drew.
These tests fill the sidebar in, press the buttons, and check that what
appears is what the command line would have said, and that a problem shows as
a sentence rather than a traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import make_counts, write_split

import preffect.gui
from preffect.cli import main

pytestmark = pytest.mark.extras

PAGE = Path(preffect.gui.__file__)


@pytest.fixture
def trained(tmp_path):
    folder = write_split(tmp_path / "train", [make_counts(n_samples=24, n_genes=12, seed=0)])
    model = tmp_path / "m.pt"
    assert main(["train", "--train", str(folder), "--out", str(model),
                 "--epochs", "2", "--batch-size", "8", "--seed", "0", "--threads", "2"]) == 0
    return folder, model


def _page():
    testing = pytest.importorskip("streamlit.testing.v1", reason="needs the gui extra")
    return testing.AppTest.from_file(str(PAGE), default_timeout=300)


def test_the_page_opens_with_nothing_filled_in():
    at = _page().run()
    assert not at.exception
    assert "PREFFECT" in at.title[0].value


def test_checking_a_folder_shows_what_is_in_it(trained):
    folder, _ = trained
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(folder))
    at.button(key="check").click().run()
    assert not at.exception
    assert any("usable" in s.value for s in at.success)
    table = at.table[0].value
    assert int(table["samples"].iloc[0]) == 24
    assert int(table["genes"].iloc[0]) == 12


def test_checking_a_missing_folder_shows_the_sentence_not_a_traceback(tmp_path):
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(tmp_path / "absent"))
    at.button(key="check").click().run()
    assert not at.exception
    assert any("no such folder" in e.value for e in at.error)


def test_applying_a_model_draws_the_figures(trained):
    folder, model = trained
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(folder))
    at.text_input(key="model_path").set_value(str(model))
    at.button(key="apply").click().run()
    assert not at.exception, at.exception
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics["samples"] == "24"
    assert metrics["genes matched"] == "12"
    assert metrics["likelihood"] == "NB"
    headings = [m.value for m in at.markdown]
    assert any("Library sizes" in h for h in headings)
    assert any("Observed against fitted" in h for h in headings)
    assert len(at.download_button) == 1


def test_the_page_says_which_distribution_the_model_was_fitted_with(tmp_path):
    """The page applies a saved model and does not fit one, so the choice is
    `preffect train --likelihood`. What it owes the reader is which choice was
    made: a zero-inflated model and a plain one of the same kind look the same
    here, and they predict different counts."""
    folder = write_split(tmp_path / "train", [make_counts(n_samples=24, n_genes=12, seed=0)])
    model = tmp_path / "zinb.pt"
    assert main(["train", "--train", str(folder), "--out", str(model),
                 "--likelihood", "ZINB", "--epochs", "2", "--batch-size", "8",
                 "--seed", "0", "--threads", "2"]) == 0

    at = _page().run()
    at.text_input(key="data_folder").set_value(str(folder))
    at.text_input(key="model_path").set_value(str(model))
    at.button(key="apply").click().run()
    assert not at.exception, at.exception
    assert {m.label: str(m.value) for m in at.metric}["likelihood"] == "ZINB"


def test_applying_without_a_model_says_so(trained):
    folder, _ = trained
    at = _page().run()
    at.text_input(key="data_folder").set_value(str(folder))
    at.button(key="apply").click().run()
    assert not at.exception
    assert any("saved model" in e.value for e in at.error)
