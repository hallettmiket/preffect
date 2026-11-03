# PREFFECT 0.2

PREFFECT fits a statistical model to RNA sequencing counts from formalin-fixed
paraffin-embedded tissue, so that the cleaned values can be used with ordinary
analysis tools.

Formalin-fixed paraffin-embedded tissue, usually shortened to FFPE, is how
pathology departments have preserved samples for a century. The RNA in those
blocks degrades, so sequencing them gives counts with many zeros that are not
real zeros, and occasional extreme values. PREFFECT learns a distribution for
each transcript while accounting for technical and biological differences
between samples, and can borrow information across matched tissues from the same
patient and across similar samples.

Version 0.2 is the current release, a rewrite of version 0.1. The method is
described in Mucaki *et al.*, *Journal of Translational Medicine* 23:1023
(2025), <https://doi.org/10.1186/s12967-025-07031-y>. Version 0.1, the code
that paper describes, remains available under the `v0.1` tag of this
repository.

## Why a rewrite

Version 0.1 has no tests, is not installable as a package, keeps its settings in
a dictionary that several parts of the code modify as it runs, and cannot be
installed against current versions of its dependencies. A review of it before
the rewrite found twenty-one defects, several of which change results silently.
[`CHANGELOG.md`](CHANGELOG.md) lists every fix in point form, and opens with how
to tell which of them explains a difference between your version 0.1 and
version 0.2 results.

The published results were checked against those defects and none of them was
affected. This rewrite is about the software, not the science.

## What is built so far

All three models train, save, reload and can be applied to new data, from
Python, from the command line, or from a page in the browser. The full model
joined them on 2026-11-02: it could be built, run forward and scored before
that, but its training loop stopped at the first batch (finding 55). The
suite now trains a two-tissue model from Python and runs `--kind full`
through the command line and then `apply`, on the vignette's own matched
tissues; see `tests/test_full_model.py` and `tests/test_vignettes.py`.

| Built | Status |
|---|---|
| Typed, validated settings | done |
| Finding and reading count files | done |
| Checks on the counts, and on the splits | done |
| Selecting genes and samples, masking, batching | done |
| The count distributions | done |
| Choosing between them on your own counts | done; ported 2026-11-02 (findings 57, 58 and 60) |
| The simple model | done |
| The objective | done |
| Training, checkpoints | done |
| The single model (one network) | done |
| The full model (matched tissues) | done; trains since 2026-11-02 (finding 55) |
| Patients missing one tissue | done; padded since 2026-11-02 (issue #14, finding 62) |
| Inference and imputation | done |
| Command-line interface | done |
| Clustering, figures, PAM50 heatmap | done |
| Documentation site | done |
| Graphical interface: check, apply, figures | done |
| Honest vignettes, transfer learning | done |

## Installing

There are two ways in, and which one you want depends on whether you intend to
change the code. Both install from GitHub; there is no release on the Python
Package Index.

### To run PREFFECT on your own data

One command, and no copy of the source:

```bash
uv tool install git+https://github.com/hallettmiket/preffect
preffect --version
```

That builds PREFFECT in an environment of its own and puts a `preffect` command
on your path. Nothing else on the machine changes, and there is no folder to
keep. It needs [uv](https://docs.astral.sh/uv/).

Add the optional parts by naming them in front of the address:

```bash
uv tool install "preffect[viz] @ git+https://github.com/hallettmiket/preffect"
```

To move to a newer version later, `uv tool upgrade preffect`. To remove it,
`uv tool uninstall preffect`. On a machine without a GPU, ask for PyTorch 2.6
from its CPU index at the same time; [getting started](docs_src/getting-started.md)
gives the command and the reason, which is that later releases' CPU kernels are
slower and the default download is mostly GPU libraries.

### To work on PREFFECT

Take a copy and install it in place, so that an edit to the source takes effect
at the next command:

```bash
git clone https://github.com/hallettmiket/preffect.git
cd preffect
bash scripts/install.sh
```

That makes an isolated environment in `.venv`, installs PREFFECT and its tools
into it, checks that the package imports and the command runs, and runs the
fast test loop. It uses [uv](https://docs.astral.sh/uv/) if uv is installed and
the standard library's `venv` and `pip` if it is not; it will not download an
installer from the internet to get uv. `--minimal` skips the optional extras,
`--force` replaces an existing environment, `--help` lists the rest.

This route deliberately leaves the command inside the environment rather than
on your path, so that the copy you are editing cannot be confused with one
installed elsewhere:

```bash
.venv/bin/preffect --help                    # what the command line can do
.venv/bin/python -m pytest -m 'not extras'   # the fast loop, about 20 s
.venv/bin/python -m pytest                   # everything, about 2 min
```

## Documentation

```bash
.venv/bin/mkdocs serve      # preview on localhost:8000
.venv/bin/mkdocs build      # writes site/, which is not committed
```

The API reference is generated from the docstrings, so there is no second copy
to go stale. Version 0.1 committed its built HTML without the sources, which
means its documentation cannot be rebuilt or corrected.

Five extras, kept apart so that fitting a model does not drag in a plotting
stack: `graph` for the single and full models, `cluster` for grouping samples
and scoring the grouping, `viz` for figures, `docs` for the site, `gui` for
the browser interface.

Python 3.11 or later is required. PyTorch 2.6 or later is required, which is a
floor rather than a preference: version 2.6 changed how saved files are read,
and version 0.1's saved models cannot be loaded by it at all.

## Using it from the command line

```bash
preffect example-data --out example_data                 # a small simulated cohort to try these on
preffect check --train path/to/train                     # look before you run
preffect train --train path/to/train --out model.pt --epochs 500 --seed 0
preffect apply --model model.pt --data path/to/new --out cleaned.h5ad
```

`check` reads the counts without fitting anything and says whether they can be
used, including whether a validation set you supplied is really the training
set again.

One flag worth knowing: `--threads` defaults to 4, which suits a small example.
PyTorch otherwise uses one thread per core, which for this work is slower, not
faster. Measured on 528 samples by 18,790 genes, 16 threads was fastest at 1.2
seconds an epoch, and 32 was slower. Every command and every option is on
[the command line page](docs_src/command-line.md).

## Settings

Settings are grouped by what they affect, and every group is checked when it is
built. A setting that cannot be honoured stops the run with a message naming it,
rather than being quietly changed into something else.

```python
from pathlib import Path
from preffect import DataConfig, ModelConfig, ModelKind, PreffectConfig

config = PreffectConfig(
    data=DataConfig(
        train_path=Path("/path/to/train"),
        validation_path=Path("/path/to/validation"),   # optional
    ),
    model=ModelConfig(kind=ModelKind.SIMPLE, latent_dim=20),
)
```

Two things worth knowing:

- **A validation set is optional.** In version 0.1 a validation folder had to
  exist before anything would run, so cohorts too small to split used a copy of
  the training data and the reported validation loss was really a training loss.
  Here, if you do not supply one, no validation loss is computed and none is
  reported.
- **Settings cannot be changed after they are built.** To vary one, call
  `config.replace(...)`, which returns a new set of settings and re-checks all
  of them.

## Repository layout

```
src/preffect/   the package
tests/          tests, most guarding against a specific past defect
docs_src/       the user documentation, which mkdocs builds into docs/ from these pages and the docstrings
docs/           the built documentation site, served at https://mikehallett.science/preffect/
vignettes/      a notebook that walks the worked example, and the script behind `preffect example-data`
CHANGELOG.md    every fix and change since version 0.1
```

The development repository additionally carries `decisions/`, one dated file
per decision with who decided and why; `exp/`, experiments that settle an open
question; `studies/`, analyses of the published results; `tools/`, scripts run
by hand; and the full record of every defect found. Those are the working
record, not part of a release.

Data does not live in this repository. Keep inputs and outputs in a folder of
their own, outside the working tree.

## Authors

- **Eliseos J. Mucaki**, Department of Biochemistry, Western University, London, Canada
- **Wenhan Zhang**, Department of Biochemistry, Western University, London, Canada
- **Aryamaan Saha**, Department of Computer Science, Columbia University, New York, USA
- **Sabina Trebinjac**, Department of Radiation Oncology and Department of Laboratory Medicine and Pathobiology, Sunnybrook Health Sciences Centre, University of Toronto, Toronto, Canada
- **Sharon Nofech-Mozes**, Department of Laboratory Medicine and Pathobiology and Department of Anatomic Pathology, Sunnybrook Health Sciences Centre, University of Toronto, Toronto, Canada
- **Eileen Rakovitch**, Department of Radiation Oncology and Department of Laboratory Medicine and Pathobiology, Sunnybrook Health Sciences Centre, University of Toronto, Toronto, Canada
- **Vanessa Dumeaux**, Department of Anatomy and Cell Biology and Department of Oncology, Western University, London, Canada
- **Michael T. Hallett**, Department of Biochemistry and Department of Oncology, Western University, London, Canada. Contact: <michael.hallett@uwo.ca>

<p align="center">
  <img src="docs_src/assets/logos/western_logo.png" alt="Western University" height="60"/>&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs_src/assets/logos/sunnybrook_logo_2024.png" alt="Sunnybrook Health Sciences Centre" height="60"/>&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs_src/assets/logos/nserc_2024.png" alt="NSERC" height="60"/>&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs_src/assets/logos/cihr_color_logo_2024.png" alt="CIHR" height="60"/>
</p>

## Citation

Mucaki EJ, Zhang W, Saha A, Trebinjac S, Nofech-Mozes S, Rakovitch E, Dumeaux V,
Hallett MT. Generative and integrative modeling for transcriptomics with
formalin fixed paraffin embedded material. *Journal of Translational Medicine*
23, 1023 (2025). <https://doi.org/10.1186/s12967-025-07031-y>

## Licence

CC0 1.0 Universal, as version 0.1 was: the authors have dedicated this software
to the public domain, and the full text is in [`LICENSE`](LICENSE). Version 0.2
contains no copied third-party code. The count distributions follow the
parameterisation used by scvi-tools and say so in their docstring; they were
written from the definition and checked numerically, not copied.
