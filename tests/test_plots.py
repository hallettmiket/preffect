"""Diagnostic figures.

Figures are hard to test, so these check the things that actually went wrong in
version 0.1 rather than asserting on pixels: that a figure is reproducible, that
it refuses rather than drawing something misleading, and that it does not depend
on the order the genes happen to be in.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib", reason="needs the viz extra")

from preffect.config import Likelihood, ModelConfig  # noqa: E402
from preffect.data.tensors import TensorData  # noqa: E402
from preffect.errors import PreffectError  # noqa: E402
from preffect.inference import infer  # noqa: E402
from preffect.model import SimpleModel  # noqa: E402

#: Every test here needs scanpy, matplotlib or umap. See the marker's note
#: in pyproject.toml: the import cost is about 80 seconds and dwarfs the
#: work, so the edit-run loop uses -m 'not extras'.
pytestmark = pytest.mark.extras

N_SAMPLES, N_GENES = 40, 20
GENE_NAMES = [f"GENE{i}" for i in range(N_GENES)]


@pytest.fixture
def counts() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 80, (N_SAMPLES, N_GENES)).float()


@pytest.fixture
def fitted(counts):
    torch.manual_seed(0)
    model = SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=5))
    return infer(model, TensorData(inputs=counts.clone(), targets=counts))


@pytest.fixture
def fitted_zinb(counts):
    torch.manual_seed(0)
    model = SimpleModel(
        N_GENES, ModelConfig(hidden_dim=16, latent_dim=5, likelihood=Likelihood.ZINB)
    )
    return infer(model, TensorData(inputs=counts.clone(), targets=counts))


# --------------------------------------------------------------------------
# Every figure draws, and none of them touch the filesystem.
# --------------------------------------------------------------------------


def test_every_figure_draws(fitted, fitted_zinb, counts):
    from preffect import plots

    batches = np.repeat([0, 1, 2], [14, 13, 13])
    assert len(plots.library_size_histogram(fitted, counts).axes) == 2
    assert len(plots.gene_scatter(fitted, counts, gene_names=GENE_NAMES).axes) == 12
    assert len(plots.batch_adjustment(fitted, counts, batches).axes) == 6
    assert len(plots.zero_fraction_against_pi(fitted_zinb, counts).axes) == 1
    assert len(plots.library_size_and_dispersion(fitted, counts).axes) == 4


def test_drawing_writes_nothing(fitted, counts, tmp_path, monkeypatch):
    """A figure function must not save. Version 0.1's did, to a baked-in path."""
    from preffect import plots

    monkeypatch.chdir(tmp_path)
    plots.library_size_histogram(fitted, counts)
    plots.library_size_and_dispersion(fitted, counts)
    assert list(tmp_path.iterdir()) == []


def test_saving_is_a_separate_step_and_makes_its_folder(fitted, counts, tmp_path):
    from preffect import plots

    figure = plots.library_size_histogram(fitted, counts)
    target = tmp_path / "new" / "deeper" / "libsize.png"
    plots.save_figure(figure, target)
    assert target.is_file() and target.stat().st_size > 0


# --------------------------------------------------------------------------
# Genes are named, not counted.
#
# Version 0.1 drew `for x in range(0, 50)`, which is column positions 0 to 49.
# Gene order is whatever the file or the gene selection produced, so the figure
# showed a different set of genes whenever the order changed, and it raised
# IndexError on any dataset with fewer than 50 genes.
# --------------------------------------------------------------------------


def test_genes_can_be_chosen_by_name(fitted, counts):
    from preffect import plots

    figure = plots.gene_scatter(
        fitted, counts, gene_names=GENE_NAMES, genes=["GENE3", "GENE11"]
    )
    assert [axis.get_title() for axis in figure.axes] == ["GENE3", "GENE11"]


def test_the_same_genes_are_drawn_whatever_order_the_columns_are_in(fitted, counts):
    """The property version 0.1 lacked, and the reason the default is a rule."""
    from preffect import plots

    order = np.random.default_rng(0).permutation(N_GENES)
    shuffled_counts = counts[:, order]
    shuffled_names = [GENE_NAMES[i] for i in order]

    torch.manual_seed(0)
    model = SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=5))
    shuffled_fit = infer(
        model, TensorData(inputs=shuffled_counts.clone(), targets=shuffled_counts)
    )

    wanted = ["GENE3", "GENE11"]
    first = plots.gene_scatter(fitted, counts, gene_names=GENE_NAMES, genes=wanted)
    second = plots.gene_scatter(
        shuffled_fit, shuffled_counts, gene_names=shuffled_names, genes=wanted
    )
    assert [a.get_title() for a in first.axes] == [a.get_title() for a in second.axes]


def test_fewer_genes_than_the_default_is_fine(counts):
    """Version 0.1 raised IndexError below 50 genes. The real data has 19,000,
    but a vignette does not, and that is exactly where someone looks first."""
    from preffect import plots

    torch.manual_seed(0)
    small = counts[:, :4]
    model = SimpleModel(4, ModelConfig(hidden_dim=8, latent_dim=3))
    result = infer(model, TensorData(inputs=small.clone(), targets=small))

    figure = plots.gene_scatter(result, small, gene_names=GENE_NAMES[:4], n_genes=12)
    drawn = [axis for axis in figure.axes if axis.get_title()]
    assert len(drawn) == 4


def test_an_unknown_gene_name_is_refused_by_name(fitted, counts):
    from preffect import plots

    with pytest.raises(PreffectError, match="NOT_A_GENE"):
        plots.gene_scatter(
            fitted, counts, gene_names=GENE_NAMES, genes=["GENE1", "NOT_A_GENE"]
        )


def test_selecting_by_name_without_names_is_refused(fitted, counts):
    from preffect import plots

    with pytest.raises(PreffectError, match="gene_names"):
        plots.gene_scatter(fitted, counts, genes=["GENE1"])


# --------------------------------------------------------------------------
# The UMAP is seeded.
#
# Version 0.1 built umap.UMAP(metric="cosine", n_neighbors=30) with no
# random_state, in the one figure that goes in a paper, while seeding other
# UMAP calls in the same tree. The same model and data drew a different picture
# every run.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_same_latent_space_draws_the_same_umap_twice(fitted):
    from preffect import plots

    first = plots.latent_umap(fitted, seed=0)
    second = plots.latent_umap(fitted, seed=0)
    np.testing.assert_allclose(
        first.axes[0].collections[0].get_offsets(),
        second.axes[0].collections[0].get_offsets(),
    )


@pytest.mark.slow
def test_a_different_seed_gives_a_different_picture(fitted):
    """So the seed is doing something, rather than being accepted and ignored."""
    from preffect import plots

    first = plots.latent_umap(fitted, seed=0).axes[0].collections[0].get_offsets()
    second = plots.latent_umap(fitted, seed=7).axes[0].collections[0].get_offsets()
    assert not np.allclose(np.asarray(first), np.asarray(second))


@pytest.mark.slow
def test_a_cohort_smaller_than_the_neighbour_count_still_draws(counts):
    """Version 0.1 hardcoded 30 neighbours and failed on smaller cohorts."""
    from preffect import plots

    torch.manual_seed(0)
    small = counts[:8]
    model = SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=4))
    result = infer(model, TensorData(inputs=small.clone(), targets=small))
    assert plots.latent_umap(result, n_neighbors=30, seed=0) is not None


@pytest.mark.slow
def test_the_umap_says_it_is_a_umap_and_which_seed(fitted):
    """A UMAP's axes have no units and the gaps between clumps mean nothing."""
    from preffect import plots

    title = plots.latent_umap(fitted, seed=3).axes[0].get_title()
    assert "UMAP" in title and "seed 3" in title


@pytest.mark.slow
def test_mismatched_colour_labels_are_refused(fitted):
    from preffect import plots

    with pytest.raises(PreffectError, match="have to match"):
        plots.latent_umap(fitted, colour_by=np.zeros(3), seed=0)


def test_a_umap_of_two_samples_is_refused(counts):
    from preffect import plots

    torch.manual_seed(0)
    tiny = counts[:2]
    model = SimpleModel(N_GENES, ModelConfig(hidden_dim=8, latent_dim=3))
    result = infer(model, TensorData(inputs=tiny.clone(), targets=tiny))
    with pytest.raises(PreffectError, match="at least 3 samples"):
        plots.latent_umap(result, seed=0)


# --------------------------------------------------------------------------
# Refusing beats drawing something misleading.
# --------------------------------------------------------------------------


def test_the_zero_probability_figure_refuses_a_model_that_has_none(fitted, counts):
    """The plain likelihood has no zero probability, so there is nothing to plot.

    Version 0.1's habit was to produce something anyway, which is the single
    most expensive thing in it.
    """
    from preffect import plots

    with pytest.raises(PreffectError, match="ZINB"):
        plots.zero_fraction_against_pi(fitted, counts)


def test_batch_comparison_refuses_a_single_batch(fitted, counts):
    from preffect import plots

    with pytest.raises(PreffectError, match="at least two batches"):
        plots.batch_adjustment(fitted, counts, np.zeros(N_SAMPLES, dtype=int))


def test_batch_comparison_is_capped_so_the_figure_stays_readable(fitted, counts):
    """Pairs grow as the square of the batch count."""
    from preffect import plots

    many = np.arange(N_SAMPLES) % 8  # eight batches, 28 pairs if uncapped
    figure = plots.batch_adjustment(fitted, counts, many)
    assert len(figure.axes) == 2 * (plots.MAX_BATCHES * (plots.MAX_BATCHES - 1) // 2)


def test_mismatched_batch_labels_are_refused(fitted, counts):
    from preffect import plots

    with pytest.raises(PreffectError, match="have to match"):
        plots.batch_adjustment(fitted, counts, np.zeros(3, dtype=int))


def test_mismatched_counts_are_refused(fitted, counts):
    from preffect import plots

    with pytest.raises(PreffectError, match="have to match"):
        plots.gene_scatter(fitted, counts[:, :5], gene_names=GENE_NAMES[:5])


# --------------------------------------------------------------------------
# Loss curves.
# --------------------------------------------------------------------------


def test_loss_curves_draw_training_and_validation_together(counts):
    """One axes, because the gap between them is the thing worth seeing."""
    from pathlib import Path

    from preffect import plots
    from preffect.config import (
        DataConfig,
        LossWeights,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.training import train

    torch.manual_seed(0)
    model = SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=5))
    data = TensorData(inputs=counts.clone(), targets=counts)
    config = PreffectConfig(
        data=DataConfig(train_path=Path("/nowhere")),
        model=model.config,
        loss=LossWeights(),
        train=TrainConfig(epochs=5, batch_size=10),
        runtime=RuntimeConfig(seed=0, num_threads=2),
    )
    history = train(model, data, config, validation_data=data)

    figure = plots.loss_curves(history)
    assert len(figure.axes) == 2
    objective = figure.axes[0]
    assert len(objective.lines) == 2, "training and validation on the same axes"
    assert {text.get_text() for text in objective.get_legend().get_texts()} == {
        "training",
        "validation",
    }


def test_loss_curves_work_without_a_validation_set(counts):
    from pathlib import Path

    from preffect import plots
    from preffect.config import (
        DataConfig,
        LossWeights,
        PreffectConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from preffect.training import train

    torch.manual_seed(0)
    model = SimpleModel(N_GENES, ModelConfig(hidden_dim=16, latent_dim=5))
    data = TensorData(inputs=counts.clone(), targets=counts)
    history = train(
        model,
        data,
        PreffectConfig(
            data=DataConfig(train_path=Path("/nowhere")),
            model=model.config,
            loss=LossWeights(),
            train=TrainConfig(epochs=4, batch_size=10),
            runtime=RuntimeConfig(seed=0, num_threads=2),
        ),
    )
    figure = plots.loss_curves(history)
    assert len(figure.axes[0].lines) == 1


# --------------------------------------------------------------------------
# The PAM50 subtype heatmap.
#
# Finding 33, three defects in version 0.1's create_PAM50_heatmap:
#   - it matched a list of gene symbols only, and three of the nine datasets
#     name their genes with Ensembl identifiers, so it matched nothing, printed
#     "PAM50 Genes Found: 0" and drew an empty heatmap
#   - it indexed a hardcoded list of five colours by subtype, so a dataset with
#     six categories raised IndexError. Two of the nine have six.
#   - it sorted samples by the colour name it had assigned, so the groups came
#     out in the alphabetical order of five colour names
# --------------------------------------------------------------------------


def _subtype_fixture(n_per_group=6, naming="symbol"):
    """Counts carrying a real subtype signal, named either way."""
    from preffect.gene_sets import PAM50

    groups = ["Normal", "LumA", "LumB", "Her2", "Basal"]
    rng = np.random.default_rng(0)
    subtypes = np.repeat(groups, n_per_group)
    names = list(PAM50.symbols if naming == "symbol" else PAM50.ensembl)
    values = rng.normal(100, 5, size=(len(subtypes), len(names)))
    # Give each group its own raised block, so grouping is visible.
    for index, group in enumerate(groups):
        block = slice(index * 10, index * 10 + 10)
        values[subtypes == group, block] += 60
    return values, names, subtypes


def test_the_heatmap_draws_and_reports_what_it_matched():
    from preffect import plots

    values, names, subtypes = _subtype_fixture()
    figure = plots.subtype_heatmap(values, names, subtypes)
    title = figure.axes[1].get_title()
    assert "50 genes matched by symbol" in title


def test_genes_are_found_whether_named_as_symbols_or_ensembl_ids():
    """Version 0.1 only ever matched symbols, and three datasets use Ensembl."""
    from preffect import plots

    for naming in ("symbol", "ensembl"):
        values, names, subtypes = _subtype_fixture(naming=naming)
        title = plots.subtype_heatmap(values, names, subtypes).axes[1].get_title()
        assert f"matched by {naming}" in title


def test_too_few_matching_genes_is_refused_not_drawn_empty():
    """Version 0.1 printed "PAM50 Genes Found: 0" and carried on."""
    from preffect import plots

    values = np.random.default_rng(0).normal(100, 5, size=(10, 6))
    names = [f"NOT_A_PAM50_GENE_{i}" for i in range(6)]
    subtypes = np.repeat(["LumA", "Basal"], 5)

    with pytest.raises(PreffectError, match="below min_genes"):
        plots.subtype_heatmap(values, names, subtypes)


def test_the_refusal_says_what_the_gene_names_look_like():
    """So the reader can see at a glance that they passed the wrong thing."""
    from preffect import plots

    values = np.random.default_rng(0).normal(100, 5, size=(10, 3))
    subtypes = np.repeat(["LumA", "Basal"], 5)
    with pytest.raises(PreffectError, match="ENSG00000999999"):
        plots.subtype_heatmap(values, ["ENSG00000999999", "b", "c"], subtypes)


def test_six_subtypes_do_not_raise_index_error():
    """Version 0.1 indexed a five-colour list. our_dcis has six categories
    (an explicit NaN group) and GSE209998 has six (Claudin)."""
    from preffect import plots

    values, names, subtypes = _subtype_fixture()
    with_sixth = np.where(np.arange(len(subtypes)) % 7 == 0, "Claudin", subtypes)
    assert len(set(with_sixth)) == 6

    figure = plots.subtype_heatmap(values, names, with_sixth)
    entries = {t.get_text().split(" (")[0] for t in figure.axes[1].get_legend().get_texts()}
    assert "Claudin" in entries and len(entries) == 6


def test_many_more_subtypes_than_colours_still_works():
    from preffect import plots

    values, names, _ = _subtype_fixture()
    many = np.array([f"group_{i % 12}" for i in range(values.shape[0])])
    figure = plots.subtype_heatmap(values, names, many)
    assert len(figure.axes[1].get_legend().get_texts()) == 12


def test_samples_are_ordered_by_subtype_not_by_colour_name():
    """The known subtypes come out in SUBTYPE_ORDER, whatever colours are used."""
    from preffect import plots

    values, names, subtypes = _subtype_fixture()
    figure = plots.subtype_heatmap(values, names, subtypes)
    drawn = [t.get_text().split(" (")[0] for t in figure.axes[1].get_legend().get_texts()]
    assert drawn == ["Normal", "LumA", "LumB", "Her2", "Basal"]


def test_an_unknown_subtype_goes_after_the_known_ones():
    from preffect import plots

    values, names, subtypes = _subtype_fixture()
    with_unknown = np.where(np.arange(len(subtypes)) % 9 == 0, "zzz_unknown", subtypes)
    figure = plots.subtype_heatmap(values, names, with_unknown)
    drawn = [t.get_text().split(" (")[0] for t in figure.axes[1].get_legend().get_texts()]
    assert drawn[-1] == "zzz_unknown"


def test_the_subtype_order_can_be_set():
    from preffect import plots

    values, names, subtypes = _subtype_fixture()
    wanted = ["Basal", "Her2", "LumB", "LumA", "Normal"]
    figure = plots.subtype_heatmap(values, names, subtypes, subtype_order=wanted)
    drawn = [t.get_text().split(" (")[0] for t in figure.axes[1].get_legend().get_texts()]
    assert drawn == wanted


def test_every_normalisation_draws_and_an_unknown_one_is_refused():
    from preffect import plots

    values, names, subtypes = _subtype_fixture()
    for how in ("z_score", "log_z_score", "min_max"):
        assert plots.subtype_heatmap(values, names, subtypes, normalise=how) is not None
    with pytest.raises(PreffectError, match="log_z_score"):
        plots.subtype_heatmap(values, names, subtypes, normalise="quantile")


def test_a_gene_with_no_variation_does_not_produce_not_a_number():
    """Dividing by a zero standard deviation would fill the figure with NaN."""
    from preffect import plots

    values, names, subtypes = _subtype_fixture()
    values[:, 3] = 7.0  # identical in every sample
    figure = plots.subtype_heatmap(values, names, subtypes)
    assert np.isfinite(figure.axes[1].get_images()[0].get_array()).all()


def test_mismatched_subtype_labels_are_refused():
    from preffect import plots

    values, names, _ = _subtype_fixture()
    with pytest.raises(PreffectError, match="have to match"):
        plots.subtype_heatmap(values, names, np.array(["LumA", "Basal"]))


def test_mismatched_gene_names_are_refused():
    from preffect import plots

    values, names, subtypes = _subtype_fixture()
    with pytest.raises(PreffectError, match="have to match"):
        plots.subtype_heatmap(values, names[:10], subtypes)


# --------------------------------------------------------------------------
# The gene set itself.
# --------------------------------------------------------------------------


def test_the_two_namings_are_paired_and_the_same_length():
    """Version 0.1 kept them as two separate lists with nothing tying them."""
    from preffect.gene_sets import PAM50

    assert len(PAM50) == 50
    assert len(PAM50.symbols) == len(PAM50.ensembl) == 50
    assert len(set(PAM50.symbols)) == 50, "a repeated symbol would silently drop a gene"
    assert len(set(PAM50.ensembl)) == 50
    assert all(identifier.startswith("ENSG") for identifier in PAM50.ensembl)


def test_matching_picks_whichever_naming_fits_better():
    from preffect.gene_sets import PAM50, match_genes

    mostly_symbols = [*PAM50.symbols[:30], *PAM50.ensembl[:2], "other"]
    _, found, naming = match_genes(PAM50, mostly_symbols)
    assert naming == "symbol" and len(found) == 30


def test_matching_nothing_says_so_rather_than_guessing():
    from preffect.gene_sets import PAM50, match_genes

    positions, found, naming = match_genes(PAM50, ["a", "b", "c"])
    assert positions == [] and found == [] and naming == "none"


def test_the_heatmap_labels_match_the_columns_it_drew():
    """Finding 37: reversing every PAM50 label left all 48 tests green.

    The figure then draws EGFR's counts under the label UBE2C. Nothing caught
    it because `_subtype_fixture` hands the genes to the heatmap in exactly the
    panel's own order, so a label list and a column list that have drifted
    apart still line up. Here the data is in a different order from the panel,
    which is the normal case for real data, and each drawn column is checked
    against the gene its label names.
    """
    import numpy as np

    from preffect import plots
    from preffect.gene_sets import PAM50

    values, names, subtypes = _subtype_fixture()

    # Shuffle the genes, as any real dataset would have them.
    order = np.random.default_rng(5).permutation(len(names))
    shuffled_values = values[:, order]
    shuffled_names = [names[i] for i in order]

    figure = plots.subtype_heatmap(
        shuffled_values, shuffled_names, subtypes, normalise="z_score"
    )
    main = figure.axes[1]
    drawn_labels = [tick.get_text() for tick in main.get_xticklabels()]
    drawn = main.get_images()[0].get_array()

    assert set(drawn_labels) <= set(PAM50.symbols)
    assert len(drawn_labels) == drawn.shape[1]

    # The column under each label must be that gene's own values, z-scored.
    lookup = {name: i for i, name in enumerate(shuffled_names)}
    for column, label in enumerate(drawn_labels):
        source = shuffled_values[:, lookup[label]]
        expected = (source - source.mean()) / (source.std() or 1.0)
        # Rows are reordered by subtype, so compare as sets of values.
        np.testing.assert_allclose(
            np.sort(drawn[:, column]), np.sort(expected), rtol=1e-4, atol=1e-4,
            err_msg=f"column {column} is labelled {label} but holds another gene",
        )
