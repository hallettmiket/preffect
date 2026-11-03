#!/usr/bin/env bash
#
# Install PREFFECT into a virtual environment in this folder, in one command.
#
#     bash scripts/install.sh
#
# Issue #1. Installing used to be two commands that had to be typed in the
# right order with the right flags, and getting the second one slightly wrong
# installed the package into whatever Python happened to be on the path. This
# script does the same work, checks the things that silently go wrong, and
# says what to do next.
#
# Options:
#
#     --minimal          just the package and its required dependencies
#     --no-check         skip the test run at the end
#     --force            replace an existing environment instead of reusing it
#     --python=VERSION   build on this Python, for example 3.11, rather than
#                        the newest on the path; the environment then goes in
#                        .venv-VERSION so that it can sit beside .venv
#     --venv=DIR         put the environment here instead
#     --help             print this and stop
#
# Why --python exists: the newest anndata needs Python 3.12 and allows pandas
# 3, while Python 3.11 gets anndata 0.12, which holds pandas below 3. Those are
# two library stacks that read and write files differently, continuous
# integration tests both, and a change that passes on one can fail on the
# other (finding 64). Keep both environments and run the fast test loop in
# each before pushing anything that touches how files are written or read.
#
# It uses uv if uv is installed, because uv is several times faster, and the
# standard library's venv and pip if it is not. It will not download and run
# an installer from the internet to get uv: that is a decision for the person
# running this, not for a script they ran to install one package.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=""
WANTED=""
EXTRAS="[dev,viz,docs]"
CHECK=1
FORCE=0

for argument in "$@"; do
    case "$argument" in
        --minimal)  EXTRAS="" ;;
        --no-check) CHECK=0 ;;
        --force)    FORCE=1 ;;
        --python=*) WANTED="${argument#--python=}" ;;
        --venv=*)   VENV="${argument#--venv=}" ;;
        # Print the comment block above, by shape rather than by line number:
        # every leading comment line after the shebang, stopping at the first
        # line that is not one. Line numbers would need editing every time a
        # sentence is added up there, and would not complain when they weren't.
        -h|--help)
            awk '/^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' \
                "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "install.sh: unknown option '$argument'" >&2; exit 2 ;;
    esac
done

say() { printf '\n== %s\n' "$1"; }
die() { printf '\ninstall.sh: %s\n' "$1" >&2; exit 1; }

[ -f "$REPO/pyproject.toml" ] || die "no pyproject.toml in $REPO, so this is not the PREFFECT repository"

# ---------------------------------------------------------------------------
# A Python new enough to run this. Checked before anything is created, because
# the failure otherwise arrives after the virtual environment exists and is
# reported as a dependency resolution error, which is not what went wrong.
# ---------------------------------------------------------------------------
PYTHON=""
if [ -n "$WANTED" ]; then
    # A particular version was asked for: the one on the path by that name,
    # or one uv has installed. Nothing is downloaded to satisfy it, for the
    # reason given at the top.
    if command -v "python$WANTED" >/dev/null 2>&1; then
        PYTHON="python$WANTED"
    elif command -v uv >/dev/null 2>&1 && uv python find "$WANTED" >/dev/null 2>&1; then
        PYTHON="$(uv python find "$WANTED")"
    fi
    [ -n "$PYTHON" ] || die "no Python $WANTED was found on the path or among uv's interpreters.
Install one (for example: uv python install $WANTED) and run this again."
    "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null \
        || die "Python $WANTED is older than 3.11, which PREFFECT needs."
    [ -n "$VENV" ] || VENV="$REPO/.venv-$WANTED"
else
    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
    [ -n "$PYTHON" ] || die "PREFFECT needs Python 3.11 or newer and none was found on the path.
Install one, or load a module, and run this again."
    [ -n "$VENV" ] || VENV="$REPO/.venv"
fi
SHOWN="${VENV#"$REPO"/}"

say "Python: $($PYTHON --version) at $(command -v "$PYTHON")"

# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
if [ -d "$VENV" ] && [ "$FORCE" = 1 ]; then
    say "removing the existing environment at $VENV"
    rm -rf "$VENV"
fi

if command -v uv >/dev/null 2>&1; then
    INSTALLER="uv"
    say "using uv ($(uv --version))"
else
    INSTALLER="pip"
    say "uv is not installed, using venv and pip (slower, works the same)
To use uv instead, install it and run this again: https://docs.astral.sh/uv/"
fi

if [ -d "$VENV" ]; then
    say "reusing the environment at $VENV (--force replaces it)"
elif [ "$INSTALLER" = "uv" ]; then
    uv venv --python "$PYTHON" "$VENV"
else
    "$PYTHON" -m venv "$VENV"
fi

[ -x "$VENV/bin/python" ] || die "the environment at $VENV has no python in it. Try again with --force."

# ---------------------------------------------------------------------------
# The package
# ---------------------------------------------------------------------------
say "installing preffect$EXTRAS"
if [ "$INSTALLER" = "uv" ]; then
    uv pip install --python "$VENV/bin/python" -e "$REPO$(printf '%s' "${EXTRAS:+$EXTRAS}")"
else
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install -e "$REPO$(printf '%s' "${EXTRAS:+$EXTRAS}")"
fi

# ---------------------------------------------------------------------------
# Does it work? An install that reports success and then cannot import the
# package is worse than one that fails, because it fails later and somewhere
# less obvious.
# ---------------------------------------------------------------------------
say "checking the install"

# -P stops Python putting the current directory at the front of the import
# path. Without it this check imports whatever folder named "preffect" happens
# to sit beside wherever the script was run from, which on a machine that also
# has version 0.1 checked out is version 0.1: the check reported a broken
# install of a package it had not installed. Found by this script, on itself.
"$VENV/bin/python" -P -c 'import preffect; print("preffect", preffect.__version__)' \
    || die "the package installed but cannot be imported"
"$VENV/bin/preffect" --help >/dev/null || die "the preffect command is installed but will not run"
echo "the preffect command runs"

if [ "$CHECK" = 1 ] && [ -n "$EXTRAS" ]; then
    say "running the fast test loop (about 20 seconds)"
    cd "$REPO"
    "$VENV/bin/python" -m pytest -m 'not extras' -q || die "the tests do not pass in this environment.
The install itself worked; something in it does not. Report the output above."
fi

cat <<INSTRUCTIONS

== Done. The environment is at $SHOWN

    $SHOWN/bin/preffect --help              what the command line can do
    $SHOWN/bin/python -m pytest -m 'not extras'   the fast test loop, about 20 s
    $SHOWN/bin/python -m pytest             everything, about 2 minutes
    $SHOWN/bin/mkdocs serve                 the documentation on localhost:8000

Data does not live in this repository. Keep inputs and outputs in a folder
of their own, outside the working tree.

Start with docs_src/getting-started.md, or state_of_the_project.md if you are
picking up the development work.
INSTRUCTIONS
