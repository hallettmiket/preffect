"""Finding and reading the count files for a run.

One folder holds one split. Inside it, one file per tissue, named so that it
ends in ``tau_<n>.h5ad`` where ``<n>`` is the tissue number counting from 1.
A single-tissue run has one file ending ``tau_1.h5ad``.

Two deliberate choices, both from defects in version 0.1:

**The search does not descend into subfolders.** Version 0.1 used a
non-recursive listing too, which was correct, but by accident rather than by
decision: several real dataset folders contain an ``old/`` subfolder holding
previous versions of the same files, and a recursive search would silently pick
them up. The behaviour is kept and now it is written down.

**The number of tissues found must match the number asked for.** Version 0.1
quietly changed the requested number to whatever it found, and the message it
logged about doing so printed the already-corrected value, so it was wrong in
both directions at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad

from preffect.errors import DataError

__all__ = ["TISSUE_FILE_PATTERN", "find_tissue_files", "load_tissues"]

#: A count file is recognised by the tissue number at the end of its name.
TISSUE_FILE_PATTERN = re.compile(r"tau_(\d+)\.h5ad$")


def find_tissue_files(folder: Path, n_tissues: int) -> list[Path]:
    """Return one file per tissue, ordered by tissue number.

    :param folder: the folder holding one split, for example the training set.
    :param n_tissues: how many tissues to expect.
    :raises DataError: if the folder is missing, or the files found do not
        number exactly ``n_tissues`` with tissue numbers 1 to ``n_tissues``.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise DataError(
            f"no such folder: {folder}. It should hold one '*tau_<n>.h5ad' file "
            f"per tissue."
        )

    found: dict[int, Path] = {}
    duplicates: dict[int, list[Path]] = {}
    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        match = TISSUE_FILE_PATTERN.search(entry.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in found:
            duplicates.setdefault(index, [found[index]]).append(entry)
        else:
            found[index] = entry

    if duplicates:
        detail = "; ".join(
            f"tissue {i}: {', '.join(p.name for p in paths)}"
            for i, paths in sorted(duplicates.items())
        )
        raise DataError(
            f"more than one file for the same tissue in {folder}. {detail}. "
            f"Each tissue number may appear once."
        )

    if not found:
        others = [p.name for p in sorted(folder.iterdir()) if p.is_file()][:5]
        raise DataError(
            f"no files matching '*tau_<n>.h5ad' in {folder}. "
            + (f"The folder does contain: {', '.join(others)}." if others
               else "The folder is empty.")
            + " Note that subfolders are not searched."
        )

    expected = set(range(1, n_tissues + 1))
    actual = set(found)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts = [f"{folder} was asked for {n_tissues} tissue(s)"]
        if missing:
            parts.append(f"no file for tissue number {missing}")
        if extra:
            parts.append(
                f"found tissue number {extra}, which is outside 1 to {n_tissues}"
            )
        parts.append(f"files present: {sorted(p.name for p in found.values())}")
        raise DataError(". ".join(parts) + ".")

    return [found[i] for i in range(1, n_tissues + 1)]


def load_tissues(folder: Path, n_tissues: int) -> list[ad.AnnData]:
    """Read one split from disk, ordered by tissue number.

    :raises DataError: if a file cannot be read, naming which one.
    """
    tissues = []
    for path in find_tissue_files(folder, n_tissues):
        try:
            tissues.append(ad.read_h5ad(path))
        except Exception as exc:  # noqa: BLE001 - the cause is reported verbatim
            raise DataError(f"could not read {path}: {exc}") from exc
    return tissues
