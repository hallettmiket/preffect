"""Preparing input: the network, the correspondence table, and the writer.

Preparing a file is the step a new user stumbles on, and until this module
existed the package gave them nothing but a description of the layout. These
tests cover two things. Every refusal, because a preparation step that repairs
what it is given quietly is how version 0.1's problems reached results. And
every function's result end to end: a file this writes is read by
``preffect check``, trains a model through the command line, aligns through
the full model's matching step, and gives ``load_adjacency`` the number of
edges the summary printed.

The network builder is measured against ``exp/16_cohort_sample_network``,
which is where the recipe comes from. That test reads the project's own
cohort and skips where it is not reachable; the one beside it builds a cohort
of its own and always runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from preffect.cli import main
from preffect.data.graph import ADJACENCY_KEY, load_adjacency
from preffect.data.tissues import ASSOCIATION_KEY, align_tissues
from preffect.errors import DataError, PreffectError
from preffect.prepare import (
    DEFAULT_NEIGHBOURS,
    add_network_to_file,
    correspondence_table,
    describe_sample_network,
    nearest_neighbour_network,
    read_counts_table,
    read_sample_table,
    sample_correlations,
    tissue_file_name,
    write_tissue,
)

# --------------------------------------------------------------------------
# A cohort to prepare: two groups of samples whose counts differ by group, so
# a network built from the counts has something real to find.
# --------------------------------------------------------------------------


def cohort(
    n_per_group: int = 8,
    n_genes: int = 30,
    seed: int = 0,
    prefix: str = "S",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts and a sample table for two groups of samples, by name."""
    rng = np.random.default_rng(seed)
    labels = np.array(["red"] * n_per_group + ["blue"] * n_per_group)
    means = {"red": rng.integers(5, 60, n_genes), "blue": rng.integers(5, 60, n_genes)}
    counts = np.vstack([rng.poisson(means[label]) for label in labels])
    names = [f"{prefix}{i:02d}" for i in range(len(labels))]
    counts_frame = pd.DataFrame(
        counts,
        index=pd.Index(names, dtype=object),
        columns=pd.Index([f"GENE_{j:03d}" for j in range(n_genes)], dtype=object),
    )
    samples = pd.DataFrame(
        {"group": labels, "patient": [f"P{i:02d}" for i in range(len(labels))]},
        index=pd.Index(names, dtype=object),
    )
    return counts_frame, samples


# --------------------------------------------------------------------------
# The network builder against exp/16, which is where the recipe comes from.
#
# The decision of 2026-10-30 on the cohort's sample network adopted a
# ten-neighbour network over 220 DCIS samples, 1,988 edges, 42.5 per cent of
# them joining two samples of the same PAM50 subtype against 20.2 per cent by
# chance. This has to reproduce that network edge for edge, or the function is
# not the recipe that was measured.
# --------------------------------------------------------------------------

EXPERIMENT_16 = "16_cohort_sample_network"


def _exp16_folder() -> Path | None:
    """Where exp/16 wrote its networks, or None if that tree is not reachable."""
    root = os.environ.get("MURMURENT_DATA_ROOT")
    if not root:
        return None
    folder = Path(root) / "append_only" / "preffect_development" / EXPERIMENT_16
    return folder if folder.is_dir() else None


def test_the_builder_reproduces_the_network_exp16_stored():
    """The ten-neighbour network of the decision of 2026-10-30, edge for edge."""
    folder = _exp16_folder()
    if folder is None:
        pytest.skip("the project's data root is not reachable from here")
    stored_file = folder / "network_knn_10_1.npz"
    cohort_folder = folder / "our_dcis_noblacklist_with_knn_10_1" / "train"
    if not stored_file.exists() or not cohort_folder.is_dir():
        pytest.skip(f"exp/{EXPERIMENT_16} has not been run on this machine")

    stored = np.load(stored_file, allow_pickle=True)
    adata = ad.read_h5ad(next(cohort_folder.glob("*.h5ad")))
    assert list(adata.obs_names) == list(stored["sample_names"]), (
        "the corrected cohort and the stored network are not the same samples"
    )

    network = nearest_neighbour_network(np.asarray(adata.X), DEFAULT_NEIGHBOURS)
    assert np.array_equal(network, np.asarray(stored["network"])), (
        "the network built here differs from the one exp/16 stored and the "
        "decision of 2026-10-30 adopted"
    )

    summary = describe_sample_network(network, adata.obs["subtype"].to_numpy(), "subtype")
    assert summary.samples == 220
    assert summary.edges == 1988
    assert round(summary.same_label_fraction, 3) == 0.425
    assert round(summary.same_label_by_chance, 3) == 0.202


def test_the_builder_finds_the_groups_in_a_cohort_built_here():
    """The test above needs the lab's data. This one needs nothing and runs anywhere."""
    counts, samples = cohort(n_per_group=10, seed=3)
    network = nearest_neighbour_network(counts.to_numpy(), 4)

    assert network.shape == (20, 20)
    assert np.array_equal(network, network.T), "a network is undirected"
    assert not np.diag(network).any(), "no sample is linked to itself"

    summary = describe_sample_network(network, samples["group"].to_numpy(), "group")
    assert summary.edges == int(np.count_nonzero(np.triu(network, 1)))
    assert summary.degree_min >= 4, "every sample names four others, so it has at least four"
    assert summary.same_label_fraction > summary.same_label_by_chance


# --------------------------------------------------------------------------
# What the network builder refuses.
# --------------------------------------------------------------------------


def test_no_neighbours_is_refused_because_it_leaves_no_edges():
    """A network with no edges is finding 35: hours of training over nothing."""
    counts, _ = cohort(n_per_group=5)
    with pytest.raises(DataError, match="no edges"):
        nearest_neighbour_network(counts.to_numpy(), 0)


def test_a_complete_network_is_refused():
    """Every sample linked to every other says nothing about which resemble which."""
    counts, _ = cohort(n_per_group=3)
    with pytest.raises(DataError, match="every sample to every other"):
        nearest_neighbour_network(counts.to_numpy(), 5)


def test_more_neighbours_than_samples_is_refused_by_name():
    counts, _ = cohort(n_per_group=3)
    with pytest.raises(DataError, match="6 samples"):
        nearest_neighbour_network(counts.to_numpy(), 40)


def test_a_sample_with_no_variation_is_refused_rather_than_correlated():
    """Its correlation with everything is undefined, and numpy says so in a warning."""
    counts, _ = cohort(n_per_group=4)
    counts.iloc[2] = 7
    with pytest.raises(DataError, match="same value at every gene"):
        nearest_neighbour_network(counts.to_numpy(), 3)


def test_missing_counts_are_refused():
    counts, _ = cohort(n_per_group=4)
    counts = counts.astype(float)
    counts.iloc[0, 0] = np.nan
    with pytest.raises(DataError, match="missing"):
        nearest_neighbour_network(counts.to_numpy(), 3)


def test_one_sample_cannot_have_a_network():
    with pytest.raises(DataError, match="at least two samples"):
        sample_correlations(np.array([[1, 2, 3]]))


def test_the_same_counts_give_the_same_network():
    """Ties are broken by position, so nothing about this is random."""
    counts, _ = cohort(n_per_group=6, seed=11)
    first = nearest_neighbour_network(counts.to_numpy(), 3)
    second = nearest_neighbour_network(counts.to_numpy(), 3)
    assert np.array_equal(first, second)


def test_mutual_keeps_only_links_both_samples_name():
    counts, _ = cohort(n_per_group=8, seed=5)
    union = nearest_neighbour_network(counts.to_numpy(), 3)
    mutual = nearest_neighbour_network(counts.to_numpy(), 3, mutual=True)
    assert mutual.sum() <= union.sum()
    assert np.array_equal(mutual & union, mutual), "every mutual link is a link"


def test_the_summary_counts_samples_with_no_link():
    """Mutual links can leave a sample out, and the summary has to say so."""
    network = np.zeros((4, 4), dtype=np.int8)
    network[0, 1] = network[1, 0] = 1
    summary = describe_sample_network(network)
    assert summary.edges == 1
    assert summary.isolated_samples == 2
    assert summary.same_label_fraction is None
    assert "2 sample(s) with no link at all" in summary.describe()


def test_a_summary_refuses_labels_that_are_not_one_per_sample():
    network = np.zeros((4, 4), dtype=np.int8)
    network[0, 1] = network[1, 0] = 1
    with pytest.raises(DataError, match="one label per sample"):
        describe_sample_network(network, ["a", "b"])


def test_a_summary_refuses_a_network_that_is_not_symmetric():
    network = np.zeros((3, 3), dtype=np.int8)
    network[0, 1] = 1
    with pytest.raises(DataError, match="not symmetric"):
        describe_sample_network(network)


# --------------------------------------------------------------------------
# The correspondence table, which is what the full model cannot run without.
# --------------------------------------------------------------------------


def two_tissues(missing: tuple[int, ...] = ()) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Two tissues of the same eight patients, with the named ones absent from the second."""
    first_counts, first_samples = cohort(n_per_group=4, seed=1, prefix="TUM")
    second_counts, second_samples = cohort(n_per_group=4, seed=2, prefix="NOR")
    keep = [i for i in range(len(second_samples)) if i not in missing]
    return (
        [first_counts, second_counts.iloc[keep]],
        [first_samples, second_samples.iloc[keep]],
    )


def test_a_complete_cohort_gets_one_row_per_patient_and_no_gaps():
    _, samples = two_tissues()
    table = correspondence_table(samples, "patient")
    assert list(table.columns) == ["tau_1", "tau_2"]
    assert len(table) == 8
    assert not (table == "NA").to_numpy().any()
    assert table.loc["P00", "tau_1"] == "TUM00"
    assert table.loc["P00", "tau_2"] == "NOR00"


def test_a_patient_missing_a_tissue_gets_the_word_the_reader_treats_as_missing():
    _, samples = two_tissues(missing=(1, 5))
    table = correspondence_table(samples, "patient")
    assert len(table) == 8, "the patient is kept, with a gap, not dropped"
    assert list(table["tau_2"] == "NA") == [False, True, False, False, False, True, False, False]


def test_a_patient_only_the_second_tissue_knows_about_still_gets_a_row():
    _, samples = two_tissues()
    extra = pd.DataFrame({"patient": ["P99"]}, index=pd.Index(["NOR99"], dtype=object))
    samples[1] = pd.concat([samples[1], extra])
    table = correspondence_table(samples, "patient")
    assert table.loc["P99", "tau_1"] == "NA"
    assert table.loc["P99", "tau_2"] == "NOR99"


def test_a_patient_listed_twice_in_one_tissue_is_refused():
    """Aligning on a table like that repeats one row and drops another in silence."""
    _, samples = two_tissues()
    samples[0].loc["TUM00", "patient"] = "P01"
    with pytest.raises(DataError, match="more than one sample for 1 patient"):
        correspondence_table(samples, "patient")


def test_a_missing_patient_column_is_refused_by_name():
    _, samples = two_tissues()
    with pytest.raises(DataError, match="subject"):
        correspondence_table(samples, "subject")


def test_a_sample_with_no_patient_recorded_is_refused():
    _, samples = two_tissues()
    samples[0] = samples[0].copy()
    samples[0].loc["TUM02", "patient"] = np.nan
    with pytest.raises(DataError, match="records no patient"):
        correspondence_table(samples, "patient")


def test_a_patient_called_na_is_refused_because_that_word_means_missing():
    _, samples = two_tissues()
    samples[0] = samples[0].copy()
    samples[0].loc["TUM02", "patient"] = "NA"
    with pytest.raises(DataError, match="that is the word the table uses"):
        correspondence_table(samples, "patient")


def test_no_tissues_at_all_is_refused():
    with pytest.raises(DataError, match="at least one tissue"):
        correspondence_table([], "patient")


def test_a_written_table_aligns_through_the_full_models_matching_step(tmp_path):
    """The end this exists for: the table has to be read by what reads it."""
    counts, samples = two_tissues(missing=(3,))
    table = correspondence_table(samples, "patient")
    folder = tmp_path / "full"
    for index, (tissue_counts, tissue_samples) in enumerate(
        zip(counts, samples, strict=True), start=1
    ):
        write_tissue(folder, tissue_counts, tissue_samples, tissue=index, association=table)

    tissues = [ad.read_h5ad(folder / tissue_file_name(i)) for i in (1, 2)]
    aligned = align_tissues(tissues, split="training")
    assert [t.n_obs for t in aligned.tissues] == [8, 8], "both tissues end up one row per patient"
    assert aligned.n_ghosts == [0, 1], "the patient with no second tissue is padded in"


def test_the_writer_refuses_a_table_that_never_names_one_of_its_samples(tmp_path):
    """Aligning would drop that sample without a word, so it is refused at writing."""
    counts, samples = two_tissues()
    table = correspondence_table(samples, "patient")
    table.loc["P00", "tau_1"] = "NA"
    with pytest.raises(DataError, match="never names 1 of tissue 1's samples"):
        write_tissue(tmp_path / "full", counts[0], samples[0], tissue=1, association=table)


def test_the_writer_refuses_a_table_with_no_column_for_this_tissue(tmp_path):
    counts, samples = two_tissues()
    table = correspondence_table(samples, "patient")
    with pytest.raises(DataError, match="no column called 'tau_3'"):
        write_tissue(tmp_path / "full", counts[0], samples[0], tissue=3, association=table)


# --------------------------------------------------------------------------
# The writer, and the file it produces.
# --------------------------------------------------------------------------


def test_a_written_file_passes_the_check_and_trains_a_model(tmp_path, capsys):
    """The whole point: what this writes is what the rest of the package reads."""
    counts, samples = cohort(n_per_group=10, n_genes=12)
    network = nearest_neighbour_network(counts.to_numpy(), 4)
    folder = tmp_path / "train"
    path = write_tissue(folder, counts, samples, network=network)
    assert path.name == "counts.tau_1.h5ad"

    assert main(["check", "--train", str(folder)]) == 0
    printed = capsys.readouterr().out
    assert "20 samples, 12 genes" in printed
    assert "the counts look usable" in printed

    # A single model needs the graph extra; without it the command refuses in a
    # sentence, which is right, and this test has nothing more to check.
    pytest.importorskip("torch_geometric", reason="needs the graph extra")
    out = tmp_path / "model.pt"
    assert main([
        "train", "--train", str(folder), "--out", str(out), "--kind", "single",
        "--epochs", "2", "--batch-size", "8", "--threads", "2",
    ]) == 0
    assert out.exists()


def test_the_network_reaches_the_model_with_the_edges_the_summary_reported(tmp_path):
    counts, samples = cohort(n_per_group=6)
    network = nearest_neighbour_network(counts.to_numpy(), 3)
    summary = describe_sample_network(network)
    path = write_tissue(tmp_path / "train", counts, samples, network=network)

    edges = load_adjacency(ad.read_h5ad(path))
    assert edges.shape[1] == 2 * summary.edges, "both directions of each edge"


def test_the_sample_annotations_travel_with_the_counts(tmp_path):
    counts, samples = cohort(n_per_group=5)
    path = write_tissue(tmp_path / "train", counts, samples)
    written = ad.read_h5ad(path)
    assert list(written.obs.columns) == ["group", "patient"]
    assert list(written.obs["group"]) == list(samples["group"])


def test_a_sample_table_in_another_order_is_matched_by_name_not_position(tmp_path):
    """Two files sorted differently would otherwise give a sample someone else's label."""
    counts, samples = cohort(n_per_group=5)
    shuffled = samples.iloc[::-1]
    path = write_tissue(tmp_path / "train", counts, shuffled)
    written = ad.read_h5ad(path)
    assert list(written.obs_names) == list(counts.index)
    for name in counts.index:
        assert written.obs.loc[name, "group"] == samples.loc[name, "group"]


def test_a_sample_table_describing_other_samples_is_refused(tmp_path):
    counts, samples = cohort(n_per_group=5)
    samples = samples.rename(index={"S00": "SOMEONE_ELSE"})
    with pytest.raises(DataError, match="different samples"):
        write_tissue(tmp_path / "train", counts, samples)


def test_negative_counts_are_refused_and_the_entries_named(tmp_path):
    counts, samples = cohort(n_per_group=4)
    counts = counts.astype(float)
    counts.iloc[1, 2] = -5
    with pytest.raises(DataError, match=r"negative, at S01/GENE_002=-5"):
        write_tissue(tmp_path / "train", counts, samples)


def test_counts_that_are_not_whole_numbers_are_refused_and_the_entries_named(tmp_path):
    counts, samples = cohort(n_per_group=4)
    counts = counts.astype(float)
    counts.iloc[0, 1] = 3.5
    with pytest.raises(DataError, match=r"not whole numbers, at S00/GENE_001=3.5"):
        write_tissue(tmp_path / "train", counts, samples)


def test_every_problem_with_the_counts_is_reported_at_once(tmp_path):
    """One run, one message: version 0.1 reported these to a log nobody opens."""
    counts, samples = cohort(n_per_group=4)
    counts = counts.astype(float)
    counts.iloc[0, 0] = -1
    counts.iloc[1, 1] = 0.25
    counts.iloc[2, 2] = np.nan
    with pytest.raises(DataError) as raised:
        write_tissue(tmp_path / "train", counts, samples)
    message = str(raised.value)
    assert "3 problem(s)" in message
    assert "negative" in message and "not whole numbers" in message and "missing" in message


def test_a_repeated_sample_name_is_refused(tmp_path):
    counts, samples = cohort(n_per_group=4)
    counts.index = pd.Index(["S00"] * 2 + list(counts.index[2:]), dtype=object)
    samples.index = counts.index
    with pytest.raises(DataError, match="sample name"):
        write_tissue(tmp_path / "train", counts, samples)


def test_a_repeated_gene_name_is_refused(tmp_path):
    """A gene name is the key selection matches on, so a repeat pulls two columns."""
    counts, samples = cohort(n_per_group=4)
    counts.columns = pd.Index(["GENE_000"] * 2 + list(counts.columns[2:]), dtype=object)
    with pytest.raises(DataError, match="gene name"):
        write_tissue(tmp_path / "train", counts, samples)


def test_counts_as_a_plain_array_need_gene_names(tmp_path):
    """Making names up would break every later match, so it refuses instead."""
    counts, samples = cohort(n_per_group=4)
    with pytest.raises(DataError, match="no gene names were given"):
        write_tissue(tmp_path / "train", counts.to_numpy(), samples)


def test_counts_as_a_plain_array_work_when_the_names_are_given(tmp_path):
    counts, samples = cohort(n_per_group=4)
    path = write_tissue(tmp_path / "train", counts.to_numpy(), samples, list(counts.columns))
    assert list(ad.read_h5ad(path).var_names) == list(counts.columns)


def test_counts_the_wrong_way_round_say_so(tmp_path):
    """Genes down the rows is how many pipelines write a table, and it is a common mistake."""
    counts, samples = cohort(n_per_group=4)
    with pytest.raises(DataError, match="transpose"):
        write_tissue(tmp_path / "train", counts.T.to_numpy(), samples, list(counts.index))


def test_a_sample_table_that_is_only_row_numbers_is_refused(tmp_path):
    counts, samples = cohort(n_per_group=4)
    with pytest.raises(DataError, match="set_index"):
        write_tissue(tmp_path / "train", counts, samples.reset_index(drop=True))


def test_a_list_of_names_is_enough_for_a_study_with_nothing_else_to_record(tmp_path):
    counts, _ = cohort(n_per_group=4)
    path = write_tissue(tmp_path / "train", counts, list(counts.index))
    assert list(ad.read_h5ad(path).obs_names) == list(counts.index)


def test_writing_over_a_file_is_refused_unless_asked_for(tmp_path):
    counts, samples = cohort(n_per_group=4)
    write_tissue(tmp_path / "train", counts, samples)
    with pytest.raises(DataError, match="is there already"):
        write_tissue(tmp_path / "train", counts, samples)
    assert write_tissue(tmp_path / "train", counts, samples, overwrite=True).exists()


def test_a_network_with_a_sample_linked_to_itself_is_refused(tmp_path):
    """The identity matrix is what every dataset version 0.1 shipped carries."""
    counts, samples = cohort(n_per_group=4)
    with pytest.raises(DataError, match="to themselves"):
        write_tissue(tmp_path / "train", counts, samples, network=np.eye(8, dtype=np.int8))


def test_a_network_with_no_edges_is_refused(tmp_path):
    counts, samples = cohort(n_per_group=4)
    with pytest.raises(DataError, match="no edges between different samples"):
        write_tissue(tmp_path / "train", counts, samples, network=np.zeros((8, 8), dtype=np.int8))


def test_a_network_of_the_wrong_size_is_refused(tmp_path):
    counts, samples = cohort(n_per_group=4)
    network = nearest_neighbour_network(cohort(n_per_group=5)[0].to_numpy(), 3)
    with pytest.raises(DataError, match="square with one row and column per sample"):
        write_tissue(tmp_path / "train", counts, samples, network=network)


def test_a_one_sided_network_is_refused(tmp_path):
    counts, samples = cohort(n_per_group=4)
    network = np.zeros((8, 8), dtype=np.int8)
    network[0, 1] = 1
    with pytest.raises(DataError, match="not symmetric"):
        write_tissue(tmp_path / "train", counts, samples, network=network)


def test_tissue_numbers_count_from_one():
    with pytest.raises(DataError, match="count from 1"):
        tissue_file_name(0)


# --------------------------------------------------------------------------
# Adding a network to a file somebody already has, which is the common case.
# --------------------------------------------------------------------------


def test_a_copy_is_written_and_the_file_read_is_left_alone(tmp_path):
    counts, samples = cohort(n_per_group=6)
    source = write_tissue(tmp_path / "train", counts, samples)
    before = source.read_bytes()

    written = add_network_to_file(source, neighbours=3, label_column="group")
    assert written.path != source
    assert source.read_bytes() == before, "the file read must not change"
    assert written.summary.edges > 0
    assert written.summary.same_label_fraction > written.summary.same_label_by_chance
    assert not written.replaced_an_empty_network


def test_the_copy_keeps_the_tissue_number_so_the_loader_still_finds_it(tmp_path):
    counts, samples = cohort(n_per_group=6)
    source = write_tissue(tmp_path / "train", counts, samples, tissue=2)
    written = add_network_to_file(source, neighbours=3)
    assert written.path.name == "counts_with_network.tau_2.h5ad"
    assert written.path.parent == source.parent


def test_a_file_the_loader_would_not_find_is_refused(tmp_path):
    counts, samples = cohort(n_per_group=4)
    source = write_tissue(tmp_path / "train", counts, samples)
    renamed = source.with_name("counts.h5ad")
    source.rename(renamed)
    with pytest.raises(DataError, match="does not end in 'tau_<n>.h5ad'"):
        add_network_to_file(renamed)


def test_an_identity_matrix_is_replaced_and_the_replacement_reported(tmp_path):
    """Finding 54: that entry is not a network, whatever it looks like."""
    counts, samples = cohort(n_per_group=6)
    source = write_tissue(tmp_path / "train", counts, samples)
    adata = ad.read_h5ad(source)
    adata.obsm[ADJACENCY_KEY] = np.eye(adata.n_obs, dtype=np.int8)
    source.unlink()
    adata.write_h5ad(source)

    written = add_network_to_file(source, neighbours=3)
    assert written.replaced_an_empty_network
    assert "replacing a network entry that had no edges in it" in written.describe()


def test_a_file_that_already_has_a_real_network_is_refused_unless_asked(tmp_path):
    counts, samples = cohort(n_per_group=6)
    network = nearest_neighbour_network(counts.to_numpy(), 3)
    source = write_tissue(tmp_path / "train", counts, samples, network=network)

    with pytest.raises(DataError, match="already carries a sample-sample network"):
        add_network_to_file(source, neighbours=4)
    written = add_network_to_file(source, neighbours=4, replace=True)
    assert written.summary.edges > 0


def test_a_label_the_file_does_not_have_is_refused_by_name(tmp_path):
    counts, samples = cohort(n_per_group=4)
    source = write_tissue(tmp_path / "train", counts, samples)
    with pytest.raises(DataError, match="no sample annotation called 'subtype'"):
        add_network_to_file(source, neighbours=2, label_column="subtype")


def test_writing_over_an_existing_copy_is_refused(tmp_path):
    counts, samples = cohort(n_per_group=4)
    source = write_tissue(tmp_path / "train", counts, samples)
    first = add_network_to_file(source, neighbours=2)
    with pytest.raises(DataError, match="is there already"):
        add_network_to_file(source, first.path, neighbours=2)


# --------------------------------------------------------------------------
# Reading the tables a study already has.
# --------------------------------------------------------------------------


def test_a_comma_separated_table_is_read_samples_down_the_rows(tmp_path):
    counts, _ = cohort(n_per_group=3)
    path = tmp_path / "counts.csv"
    counts.to_csv(path)
    read = read_counts_table(path)
    assert list(read.index) == list(counts.index)
    assert list(read.columns) == list(counts.columns)


def test_a_tab_separated_table_is_recognised_by_its_name(tmp_path):
    counts, _ = cohort(n_per_group=3)
    path = tmp_path / "counts.tsv"
    counts.to_csv(path, sep="\t")
    assert read_counts_table(path).shape == counts.shape


def test_a_table_of_genes_by_samples_is_turned_the_right_way_round(tmp_path):
    counts, _ = cohort(n_per_group=3)
    path = tmp_path / "counts.csv"
    counts.T.to_csv(path)
    read = read_counts_table(path, "genes-by-samples")
    assert list(read.index) == list(counts.index)
    assert np.array_equal(read.to_numpy(), counts.to_numpy())


def test_an_orientation_that_is_neither_is_refused_by_name():
    with pytest.raises(DataError, match="sideways"):
        read_counts_table("nothing.csv", "sideways")


def test_a_sample_table_takes_its_names_from_the_first_column(tmp_path):
    _, samples = cohort(n_per_group=3)
    path = tmp_path / "samples.csv"
    samples.rename_axis("sample").to_csv(path)
    read = read_sample_table(path)
    assert list(read.index) == list(samples.index)
    assert list(read.columns) == ["group", "patient"]


def test_a_sample_table_can_take_its_names_from_a_named_column(tmp_path):
    _, samples = cohort(n_per_group=3)
    path = tmp_path / "samples.csv"
    samples.rename_axis("sample").to_csv(path)
    read = read_sample_table(path, "patient")
    assert list(read.index) == list(samples["patient"])


def test_a_sample_table_without_that_column_is_refused_by_name(tmp_path):
    _, samples = cohort(n_per_group=3)
    path = tmp_path / "samples.csv"
    samples.rename_axis("sample").to_csv(path)
    with pytest.raises(DataError, match="no column called 'subject'"):
        read_sample_table(path, "subject")


# --------------------------------------------------------------------------
# The two commands.
# --------------------------------------------------------------------------


@pytest.fixture
def tables(tmp_path):
    """Counts and sample tables on disk, one pair per tissue, as a user would have them."""

    def _write(n_tissues: int = 1, missing: tuple[int, ...] = ()) -> list[tuple[Path, Path]]:
        written = []
        for index in range(1, n_tissues + 1):
            counts, samples = cohort(
                n_per_group=6, n_genes=12, seed=index, prefix=f"T{index}_"
            )
            if index > 1 and missing:
                keep = [i for i in range(len(samples)) if i not in missing]
                counts, samples = counts.iloc[keep], samples.iloc[keep]
            counts_path = tmp_path / f"counts_{index}.csv"
            samples_path = tmp_path / f"samples_{index}.csv"
            counts.to_csv(counts_path)
            samples.rename_axis("sample").to_csv(samples_path)
            written.append((counts_path, samples_path))
        return written

    return _write


def test_prepare_writes_a_file_the_check_accepts(tables, tmp_path, capsys):
    (counts_path, samples_path), = tables()
    folder = tmp_path / "study" / "train"
    assert main([
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(folder), "--neighbours", "3", "--label", "group",
    ]) == 0
    printed = capsys.readouterr().out
    assert "12 samples and 12 genes" in printed
    assert "edge(s)" in printed
    assert "the same group" in printed
    assert (folder / "counts.tau_1.h5ad").exists()
    assert main(["check", "--train", str(folder)]) == 0


def test_prepare_without_neighbours_writes_no_network(tables, tmp_path, capsys):
    (counts_path, samples_path), = tables()
    folder = tmp_path / "study" / "train"
    main([
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(folder),
    ])
    main(["check", "--train", str(folder)])
    assert "no network" in capsys.readouterr().out


def test_prepare_builds_the_correspondence_table_for_two_tissues(tables, tmp_path, capsys):
    pairs = tables(n_tissues=2, missing=(0, 1))
    folder = tmp_path / "study" / "train"
    arguments = ["prepare"]
    for counts_path, samples_path in pairs:
        arguments += ["--counts", str(counts_path), "--samples", str(samples_path)]
    arguments += ["--out", str(folder), "--patient-column", "patient"]
    assert main(arguments) == 0
    printed = capsys.readouterr().out
    assert "12 patient(s) across 2 tissues" in printed
    assert "tissue 2: 2 patient(s) have no sample here" in printed

    first = ad.read_h5ad(folder / "counts.tau_1.h5ad")
    assert ASSOCIATION_KEY in first.obsm
    assert main(["check", "--train", str(folder), "--tissues", "2"]) == 0


def test_prepare_refuses_two_tissues_without_a_patient_column(tables, tmp_path, capsys):
    pairs = tables(n_tissues=2)
    arguments = ["prepare"]
    for counts_path, samples_path in pairs:
        arguments += ["--counts", str(counts_path), "--samples", str(samples_path)]
    arguments += ["--out", str(tmp_path / "study")]
    assert main(arguments) == 2
    assert "--patient-column" in capsys.readouterr().err


def test_prepare_refuses_a_counts_table_with_no_sample_table(tables, tmp_path, capsys):
    pairs = tables(n_tissues=2)
    arguments = [
        "prepare",
        "--counts", str(pairs[0][0]),
        "--counts", str(pairs[1][0]),
        "--samples", str(pairs[0][1]),
        "--out", str(tmp_path / "study"),
    ]
    assert main(arguments) == 2
    assert "one of each per tissue" in capsys.readouterr().err


def test_prepare_refuses_to_write_over_what_is_there(tables, tmp_path, capsys):
    (counts_path, samples_path), = tables()
    folder = tmp_path / "study" / "train"
    arguments = [
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(folder),
    ]
    assert main(arguments) == 0
    assert main(arguments) == 2
    assert "already in that folder" in capsys.readouterr().err
    assert main([*arguments, "--overwrite"]) == 0


def test_prepare_refuses_a_label_the_sample_table_does_not_have(tables, tmp_path, capsys):
    (counts_path, samples_path), = tables()
    assert main([
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(tmp_path / "study"), "--neighbours", "3", "--label", "subtype",
    ]) == 2
    assert "no column called 'subtype'" in capsys.readouterr().err


def test_prepare_refuses_a_label_with_no_network_to_measure(tables, tmp_path, capsys):
    """An option that silently does nothing is how version 0.1 ran the wrong model."""
    (counts_path, samples_path), = tables()
    assert main([
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(tmp_path / "study"), "--label", "group",
    ]) == 2
    assert "--label describes the sample-sample network" in capsys.readouterr().err


def test_network_adds_one_to_a_folder_that_has_none(tables, tmp_path, capsys):
    (counts_path, samples_path), = tables()
    folder = tmp_path / "study" / "train"
    main([
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(folder),
    ])
    linked = tmp_path / "study" / "train_linked"
    assert main([
        "network", "--data", str(folder), "--out", str(linked),
        "--neighbours", "3", "--label", "group",
    ]) == 0
    printed = capsys.readouterr().out
    assert "edge(s)" in printed
    assert "the same group" in printed
    assert (linked / "counts.tau_1.h5ad").exists()

    assert main(["check", "--train", str(linked)]) == 0
    assert "a network with" in capsys.readouterr().out


def test_network_leaves_the_folder_it_read_alone(tables, tmp_path):
    (counts_path, samples_path), = tables()
    folder = tmp_path / "study" / "train"
    main([
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(folder),
    ])
    before = (folder / "counts.tau_1.h5ad").read_bytes()
    main([
        "network", "--data", str(folder), "--out", str(tmp_path / "linked"),
        "--neighbours", "3",
    ])
    assert (folder / "counts.tau_1.h5ad").read_bytes() == before


def test_network_refuses_to_write_over_what_is_there(tables, tmp_path, capsys):
    (counts_path, samples_path), = tables()
    folder = tmp_path / "study" / "train"
    main([
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(folder),
    ])
    arguments = [
        "network", "--data", str(folder), "--out", str(tmp_path / "linked"),
        "--neighbours", "3",
    ]
    assert main(arguments) == 0
    assert main(arguments) == 2
    assert "already in that folder" in capsys.readouterr().err


def test_network_refuses_a_number_of_neighbours_that_says_nothing(tables, tmp_path, capsys):
    (counts_path, samples_path), = tables()
    folder = tmp_path / "study" / "train"
    main([
        "prepare", "--counts", str(counts_path), "--samples", str(samples_path),
        "--out", str(folder),
    ])
    assert main([
        "network", "--data", str(folder), "--out", str(tmp_path / "linked"),
        "--neighbours", "11",
    ]) == 2
    assert "every sample to every other" in capsys.readouterr().err


def test_network_refuses_a_folder_that_is_not_there(tmp_path, capsys):
    assert main([
        "network", "--data", str(tmp_path / "nowhere"), "--out", str(tmp_path / "out"),
    ]) == 2
    assert "no such folder" in capsys.readouterr().err


def test_the_refusals_are_the_packages_own_error(tmp_path):
    """Every refusal here is caught by the command line and printed as a sentence."""
    assert issubclass(DataError, PreffectError)


# --------------------------------------------------------------------------
# Finding 66: a file that cannot be read reached the reader's caller as a
# pandas traceback.
#
# ``pd.read_csv`` raises an OSError for a path that is not there and its own
# parser errors for a file that is not a table. Neither is one of this
# package's errors, so the command line, which catches only those, printed
# twenty lines of stack ending inside pandas, and the graphical interface
# showed the same. Both now get the sentence.
# --------------------------------------------------------------------------


def test_a_counts_table_that_is_not_there_is_refused_in_a_sentence(tmp_path):
    with pytest.raises(DataError, match="could not open the counts table"):
        read_counts_table(tmp_path / "absent.csv")


def test_a_sample_table_that_is_not_there_is_refused_in_a_sentence(tmp_path):
    with pytest.raises(DataError, match="could not open the sample table"):
        read_sample_table(tmp_path / "absent.csv")


def test_a_folder_given_where_a_table_was_expected_is_refused(tmp_path):
    with pytest.raises(DataError, match="could not open the counts table"):
        read_counts_table(tmp_path)


def test_a_file_that_is_not_a_table_is_refused_in_a_sentence(tmp_path):
    """An empty file reads as a table with nothing in it, which pandas
    refuses with an error of its own."""
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataError, match="as a table"):
        read_counts_table(path)


def test_the_command_line_prints_the_sentence_rather_than_a_traceback(tmp_path, capsys):
    """What the fix is for. ``main`` catches this package's errors and prints
    them; anything else it lets through as a stack."""
    assert main([
        "prepare", "--counts", str(tmp_path / "absent.csv"),
        "--samples", str(tmp_path / "absent.csv"), "--out", str(tmp_path / "out"),
    ]) != 0
    printed = capsys.readouterr()
    assert "could not open the counts table" in printed.out + printed.err
    assert "Traceback" not in printed.err
