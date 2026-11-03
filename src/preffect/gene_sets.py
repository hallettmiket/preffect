"""Named sets of genes.

Kept as data rather than as a list someone retypes, because version 0.1 had the
PAM50 set twice, once as symbols and once as Ensembl identifiers, and nothing
tied the two together. ``_utils.py`` then did this at import:

.. code-block:: python

    # this should be removed; its only used to handle a dataset with Ensembl IDs
    PAM50genes = PAM50genes_ensembl

so the same name meant two different lists depending on which module you were
in. The heatmap lived in the module that kept symbols, and the three datasets
that use Ensembl identifiers matched nothing. See finding 33.

Here the two are one table, paired row by row, and the lookup decides which
column to use from the data in front of it.
"""

from __future__ import annotations

__all__ = ["PAM50", "GeneSet", "match_genes"]

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneSet:
    """A named set of genes, with both ways of naming each one."""

    name: str
    #: ``(symbol, ensembl_id)``, paired. Either may be used by a dataset.
    members: tuple[tuple[str, str], ...]

    def __len__(self) -> int:
        return len(self.members)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(symbol for symbol, _ in self.members)

    @property
    def ensembl(self) -> tuple[str, ...]:
        return tuple(identifier for _, identifier in self.members)


def match_genes(gene_set: GeneSet, gene_names) -> tuple[list[int], list[str], str]:
    """Find a gene set in some data, whichever way the data names its genes.

    Tries symbols and Ensembl identifiers and keeps whichever matches more,
    rather than requiring the caller to know. Version 0.1 required the caller
    to know and provided a global switch in the wrong module for it.

    :returns: the column positions that matched, their names, and which
        naming was used.
    """
    names = {str(name): position for position, name in enumerate(gene_names)}

    best: tuple[list[int], list[str], str] = ([], [], "none")
    for naming, wanted in (("symbol", gene_set.symbols), ("ensembl", gene_set.ensembl)):
        positions, found = [], []
        for name in wanted:
            if name in names:
                positions.append(names[name])
                found.append(name)
        if len(positions) > len(best[0]):
            best = (positions, found, naming)
    return best


#: The PAM50 set, the fifty genes used to assign a breast tumour to an
#: intrinsic subtype. Ported from version 0.1, which held the two namings as
#: separate lists of the same length and in the same order.
PAM50 = GeneSet(
    name="PAM50",
    members=(
        ("EGFR", "ENSG00000146648"),
        ("CDH3", "ENSG00000062038"),
        ("PHGDH", "ENSG00000092621"),
        ("ACTR3B", "ENSG00000133627"),
        ("FOXC1", "ENSG00000054598"),
        ("MIA", "ENSG00000261857"),
        ("MYC", "ENSG00000136997"),
        ("FGFR4", "ENSG00000160867"),
        ("MDM2", "ENSG00000135679"),
        ("MLPH", "ENSG00000115648"),
        ("KRT14", "ENSG00000186847"),
        ("BCL2", "ENSG00000171791"),
        ("SFRP1", "ENSG00000104332"),
        ("KRT5", "ENSG00000186081"),
        ("KRT17", "ENSG00000128422"),
        ("SLC39A6", "ENSG00000141424"),
        ("ESR1", "ENSG00000091831"),
        ("CXXC5", "ENSG00000171604"),
        ("BLVRA", "ENSG00000106605"),
        ("FOXA1", "ENSG00000129514"),
        ("GPR160", "ENSG00000173890"),
        ("NAT1", "ENSG00000171428"),
        ("MAPT", "ENSG00000186868"),
        ("PGR", "ENSG00000082175"),
        ("BAG1", "ENSG00000107262"),
        ("TMEM45B", "ENSG00000151715"),
        ("ERBB2", "ENSG00000141736"),
        ("GRB7", "ENSG00000141738"),
        ("MMP11", "ENSG00000099953"),
        ("CDC20", "ENSG00000117399"),
        ("MKI67", "ENSG00000148773"),
        ("CCNE1", "ENSG00000105173"),
        ("CENPF", "ENSG00000117724"),
        ("NUF2", "ENSG00000143228"),
        ("EXO1", "ENSG00000174371"),
        ("KIF2C", "ENSG00000142945"),
        ("ORC6", "ENSG00000091651"),
        ("ANLN", "ENSG00000011426"),
        ("CDC6", "ENSG00000094804"),
        ("RRM2", "ENSG00000171848"),
        ("UBE2T", "ENSG00000077152"),
        ("NDC80", "ENSG00000080986"),
        ("CEP55", "ENSG00000138180"),
        ("MELK", "ENSG00000165304"),
        ("TYMS", "ENSG00000176890"),
        ("CCNB1", "ENSG00000134057"),
        ("BIRC5", "ENSG00000089685"),
        ("MYBL2", "ENSG00000101057"),
        ("PTTG1", "ENSG00000164611"),
        ("UBE2C", "ENSG00000175063"),
    ),
)
