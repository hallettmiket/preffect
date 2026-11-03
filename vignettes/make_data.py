"""Write the example cohort: the same thing as ``preffect example-data``.

Kept as a script so that a copy of the source can build the data without
installing the command, and so that the tests can run the generator the way a
person would. Everything it does lives in :mod:`preffect.example_data`, and
the history of why (version 0.1's ``test/`` folders being copies of
``validation/``, issue #11) is in that module's docstring.

Run::

    .venv/bin/python vignettes/make_data.py --out vignette_data
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse the arguments, write the data, print what was written."""
    from preffect.errors import PreffectError
    from preffect.example_data import write_example_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("vignette_data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--unmatched",
        type=float,
        default=0.0,
        help="share of patients whose second tissue is removed, so the full "
        "model has a cohort with gaps to pad; zero, the default, draws no "
        "random numbers and leaves every file byte-identical",
    )
    args = parser.parse_args()
    try:
        written = write_example_data(args.out, seed=args.seed, unmatched=args.unmatched)
    except PreffectError as error:
        raise SystemExit(str(error)) from error
    print(written.describe())


if __name__ == "__main__":
    main()
