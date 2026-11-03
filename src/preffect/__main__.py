"""Make ``python -m preffect`` run the command line.

Without this file Python refuses with "No module named preffect.__main__",
and the getting-started page told people to run exactly that whenever
``preffect --version`` printed something unexpected. The one instruction
written for the case where the command is already misbehaving was itself
broken. Finding 42.

The form worth knowing is ``python -P -m preffect``: ``-P`` stops Python
putting the current folder at the front of the import path, so a folder named
``preffect`` beside you cannot be mistaken for the installed package.
"""

from preffect.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
