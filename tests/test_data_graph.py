"""Reading the sample-sample network.

This file did not exist until finding 35, which is how an empty network got all
the way to a training run at full scale without anything objecting.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from preffect.data.graph import ADJACENCY_KEY, has_adjacency, load_adjacency
from preffect.errors import DataError


def with_network(network: np.ndarray) -> ad.AnnData:
    """Counts carrying a sample-sample network of the given shape."""
    n = network.shape[0]
    adata = ad.AnnData(np.ones((n, 4), dtype=np.float32))
    adata.obsm[ADJACENCY_KEY] = network
    return adata


# --------------------------------------------------------------------------
# Reading a network that is there.
# --------------------------------------------------------------------------


def test_both_directions_of_every_edge_are_returned():
    """A graph attention layer expects both, and a network is undirected."""
    network = np.zeros((4, 4), dtype=np.int8)
    network[0, 1] = network[1, 0] = 1
    network[2, 3] = network[3, 2] = 1

    edges = load_adjacency(with_network(network), split="training")
    assert edges.shape == (2, 4)
    pairs = {tuple(pair) for pair in edges.t().tolist()}
    assert pairs == {(0, 1), (1, 0), (2, 3), (3, 2)}


def test_a_sparse_network_is_read_the_same_as_a_dense_one():
    """The real files store it as a sparse matrix."""
    from scipy.sparse import csc_matrix

    network = np.zeros((5, 5), dtype=np.int8)
    network[0, 4] = network[4, 0] = 1

    dense = load_adjacency(with_network(network), split="t")
    sparse = load_adjacency(with_network(csc_matrix(network)), split="t")
    assert dense.tolist() == sparse.tolist()


def test_self_links_are_dropped():
    """Every sample attends to itself anyway, and counting them would flatter
    any measure of how well the network is reconstructed."""
    network = np.eye(4, dtype=np.int8)
    network[0, 1] = network[1, 0] = 1

    edges = load_adjacency(with_network(network), split="training")
    assert not (edges[0] == edges[1]).any()
    assert edges.shape == (2, 2)


def test_a_missing_network_names_what_is_present():
    adata = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    adata.obsm["something_else"] = np.zeros((3, 2))
    with pytest.raises(DataError, match="something_else"):
        load_adjacency(adata, split="training")


def test_a_network_of_the_wrong_shape_is_refused():
    adata = ad.AnnData(np.ones((4, 2), dtype=np.float32))
    adata.obsm[ADJACENCY_KEY] = np.zeros((4, 3))
    with pytest.raises(DataError, match="has to be square"):
        load_adjacency(adata, split="training")


def test_has_adjacency_reports_presence_not_content():
    assert has_adjacency(with_network(np.eye(3, dtype=np.int8)))
    assert not has_adjacency(ad.AnnData(np.ones((3, 2), dtype=np.float32)))


# --------------------------------------------------------------------------
# A network with no edges is refused, not quietly accepted.
#
# Finding 35: eight of the nine real datasets carry an identity matrix under
# (finding 54: the ninth is the same identity with its rows subset)
# obsm["sample_sample_adj"]. load_adjacency correctly drops self-links, so it
# returned an empty edge list and the graph model trained on nothing: 165
# million parameters against the simple model's 14.5 million, 22.7 seconds an
# epoch against 1.2, and no neighbours to attend to. Nothing said so.
# --------------------------------------------------------------------------


def test_an_identity_network_is_refused():
    """The common case in the real data, not a rare one."""
    with pytest.raises(DataError, match="no edges between different samples"):
        load_adjacency(with_network(np.eye(6, dtype=np.int8)), split="training")


def test_the_refusal_counts_the_self_links_and_names_the_remedy():
    with pytest.raises(DataError) as raised:
        load_adjacency(with_network(np.eye(5, dtype=np.int8)), split="training")

    message = str(raised.value)
    assert "5 self-links" in message
    assert "Use the simple model" in message


def test_an_all_zero_network_is_refused_too():
    with pytest.raises(DataError, match="no edges"):
        load_adjacency(with_network(np.zeros((4, 4), dtype=np.int8)), split="training")


def test_a_single_real_edge_is_enough_to_proceed():
    """The refusal must not fire on a sparse but genuine network.

    ``our_dcis_noblacklist`` has about one edge per sample, which is sparse and
    real, and it is the only one of the nine that has any.
    """
    network = np.eye(6, dtype=np.int8)  # self-links, which are dropped
    network[0, 3] = network[3, 0] = 1  # and one real edge
    edges = load_adjacency(with_network(network), split="training")
    assert edges.shape == (2, 2), "both directions of the one edge"


# --------------------------------------------------------------------------
# A row subset of an identity is not a sparse network, and repairing its shape
# does not make it one.
#
# Finding 54: the one dataset the project believed had a network stores the
# identity matrix of a larger cohort with some rows removed and the columns
# kept. load_adjacency refuses the shape, correctly, and exp/10 "repaired" it
# by dropping the columns past the last sample, dropping the diagonal and
# reading the rest in both directions, which turns each sample's 1 into an
# edge to whichever sample now sits at its old column: 201 edges, every one a
# shift of one to nine positions, that two experiments and a decision then
# trained on. exp/16 diagnoses the shape before anything is built from it.
# --------------------------------------------------------------------------


def _exp16():
    """exp/16's diagnosis, loaded by path because every experiment is run_all.py."""
    import importlib.util
    import sys
    from pathlib import Path

    experiments = Path(__file__).resolve().parents[1] / "exp"
    if not experiments.is_dir():
        pytest.skip("exp/ is part of the development record, not of a release")
    sys.path.insert(0, str(experiments))
    path = experiments / "16_cohort_sample_network" / "run_all.py"
    spec = importlib.util.spec_from_file_location("exp16_run_all", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_with_rows_removed(n: int, removed: list[int]) -> np.ndarray:
    kept = [i for i in range(n) if i not in removed]
    return np.eye(n, dtype=np.int8)[kept]


def test_a_misaligned_row_subset_of_an_identity_is_still_refused_by_the_loader():
    """The shape refusal is what made the repair necessary; it must stay."""
    stored = _identity_with_rows_removed(12, [2, 5, 9])
    adata = ad.AnnData(np.ones((9, 4), dtype=np.float32))
    adata.obsm[ADJACENCY_KEY] = stored
    with pytest.raises(DataError, match="has to be square"):
        load_adjacency(adata, split="training")


def test_a_row_subset_of_an_identity_is_diagnosed_as_edgeless():
    """exp/16's diagnosis names the shape and says there are no edges in it."""
    module = _exp16()
    stored = _identity_with_rows_removed(12, [2, 5, 9])
    diagnosis = module.diagnose_stored(stored)
    assert diagnosis["is_identity_row_subset"] and not diagnosis["is_identity"]
    assert "row subset of a 12 by 12 identity" in diagnosis["message"]
    assert "zero edges" in diagnosis["message"]
    # The kept rows 0,1,3,4,6,7,8,10,11 have their 1 at those columns, so the
    # two above the first removed row keep it on the diagonal and the shift
    # grows by one at each removed row: shifts 0,0,1,1,2,2,2,3,3.
    assert diagnosis["shift_histogram"] == [2, 2, 3, 2]

    whole = module.diagnose_stored(np.eye(12, dtype=np.int8))
    assert whole["is_identity"] and not whole["is_identity_row_subset"]
    assert "zero edges" in whole["message"]

    real = np.zeros((4, 4), dtype=np.int8)
    real[0, 1] = real[1, 0] = 1
    assert not module.diagnose_stored(real)["is_identity_row_subset"]
    assert not module.diagnose_stored(real)["is_identity"]


def test_repairing_a_row_subset_of_an_identity_produces_only_shift_artefacts():
    """exp/10's repair, applied to such a matrix, yields edges that are all shifts of the diagonal.

    Twelve samples, three removed: the nine kept have their 1 at columns
    0, 1, 3, 4, 6, 7, 8, 10, 11. Dropping the columns past the ninth sample and
    the diagonal, then symmetrising, links sample i to sample i + (removed rows
    above it), and nothing else: two 1s sit on the diagonal and go, two sit
    past the last column and go, and the five left are shifts of 1, 1, 2, 2
    and 2. A real network would have edges at distances the removal count
    cannot explain.
    """
    module = _exp16()
    stored = _identity_with_rows_removed(12, [2, 5, 9])
    artefacts = module.shift_artefacts(stored)
    assert artefacts["all_edges_are_shifts"]
    assert artefacts["largest_shift"] == 2
    assert artefacts["edges"] == 5
    assert artefacts["shift_histogram"] == [0, 2, 3]

    real = np.zeros((6, 8), dtype=np.int8)
    real[0, 5] = 1  # five apart, which two removed rows cannot produce
    assert not module.shift_artefacts(real)["all_edges_are_shifts"]
