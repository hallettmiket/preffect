"""Named gene sets, and finding them in data that names genes either way.

Finding 33: version 0.1 held the PAM50 set twice, as symbols and as Ensembl
identifiers, with nothing tying them together, and chose between them by
rebinding an imported name at module level. That switch could not be right for
every caller, and where it was wrong it silently did nothing.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from preffect.data.selection import choose_genes
from preffect.errors import DataError
from preffect.gene_sets import PAM50, match_genes


def counts_named(names) -> ad.AnnData:
    rng = np.random.default_rng(0)
    adata = ad.AnnData(rng.normal(100, 10, size=(12, len(names))).astype(np.float32))
    adata.var_names = [str(n) for n in names]
    return adata


# --------------------------------------------------------------------------
# The set itself.
# --------------------------------------------------------------------------


def test_the_two_namings_are_paired_complete_and_distinct():
    assert len(PAM50) == 50
    assert len(set(PAM50.symbols)) == 50, "a repeat would silently drop a gene"
    assert len(set(PAM50.ensembl)) == 50
    assert all(i.startswith("ENSG") for i in PAM50.ensembl)


def test_matching_works_whichever_naming_the_data_uses():
    for naming, names in (("symbol", PAM50.symbols), ("ensembl", PAM50.ensembl)):
        positions, found, detected = match_genes(PAM50, list(names))
        assert detected == naming
        assert len(found) == 50
        assert positions == list(range(50))


def test_matching_keeps_whichever_naming_fits_better():
    mixed = [*PAM50.symbols[:30], *PAM50.ensembl[:2], "other"]
    _, found, naming = match_genes(PAM50, mixed)
    assert naming == "symbol" and len(found) == 30


def test_matching_nothing_says_none_rather_than_guessing():
    positions, found, naming = match_genes(PAM50, ["a", "b", "c"])
    assert (positions, found, naming) == ([], [], "none")


# --------------------------------------------------------------------------
# Forcing a panel through gene selection.
#
# Version 0.1's force_pam50 matched an Ensembl list against symbol-named data,
# found nothing, and reordered the ranking by an empty list, which leaves it
# unchanged. No error, no warning. Three of its own PAM50 notebooks ran that
# way on their third-party arm.
# --------------------------------------------------------------------------


def test_a_gene_set_is_matched_whichever_way_the_data_names_its_genes():
    for names in (PAM50.symbols, PAM50.ensembl):
        data = counts_named([*names, "FILLER1", "FILLER2"])
        selection = choose_genes(data, n_genes=50, force_genes=PAM50)
        assert set(selection.names) == set(names)


def test_a_panel_that_matches_nothing_raises_rather_than_doing_nothing():
    """The whole point. Silence here is what cost version 0.1 three notebooks."""
    ensembl_data = counts_named([*PAM50.ensembl, "FILLER"])

    with pytest.raises(DataError, match="silently do nothing"):
        choose_genes(ensembl_data, n_genes=10, force_genes=PAM50.symbols)


def test_the_refusal_shows_what_the_data_actually_looks_like():
    data = counts_named(["ENSG00000000001", "ENSG00000000002"])
    with pytest.raises(DataError, match="ENSG00000000001"):
        choose_genes(data, n_genes=2, force_genes=("EGFR", "CDH3"))


def test_a_gene_set_passed_to_the_wrong_data_still_raises():
    data = counts_named([f"UNRELATED{i}" for i in range(20)])
    with pytest.raises(DataError, match="PAM50"):
        choose_genes(data, n_genes=5, force_genes=PAM50)


def test_some_of_a_panel_missing_is_still_fine():
    """A panel routinely names genes an assay did not measure."""
    present = list(PAM50.symbols[:30])
    data = counts_named([*present, "FILLER1", "FILLER2"])
    selection = choose_genes(data, n_genes=30, force_genes=PAM50)
    assert set(selection.names) == set(present)


def test_forcing_nothing_is_unchanged_behaviour():
    data = counts_named([f"GENE{i}" for i in range(20)])
    assert len(choose_genes(data, n_genes=5).names) == 5
    assert len(choose_genes(data, n_genes=5, force_genes=()).names) == 5


def test_forced_genes_come_first_and_survive_a_tight_budget():
    forced = list(PAM50.symbols[:5])
    data = counts_named([*forced, *[f"GENE{i}" for i in range(40)]])
    selection = choose_genes(data, n_genes=5, force_genes=PAM50)
    assert set(selection.names) == set(forced)
    assert "named gene(s) first" in selection.basis
