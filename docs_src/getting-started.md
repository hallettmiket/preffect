# Getting started

## Installing

PREFFECT needs Python 3.11 or newer. There is no release on the Python Package
Index; it installs from GitHub.

### One command, if you only want to run it

```bash
uv tool install git+https://github.com/hallettmiket/preffect
```

That builds PREFFECT in an environment of its own and puts a `preffect` command
on your path. There is no copy of the source to keep and nothing to activate.
It needs [uv](https://docs.astral.sh/uv/).

The parts used for figures and for grouping samples are optional, because
fitting a model does not need a plotting stack. Ask for them by name:

```bash
uv tool install "preffect[viz] @ git+https://github.com/hallettmiket/preffect"
```

Later, `uv tool upgrade preffect` moves you to a newer version and
`uv tool uninstall preffect` removes it.

### On a machine without a GPU

Ask for PyTorch 2.6 from PyTorch's CPU index at the same time:

```bash
uv tool install "preffect @ git+https://github.com/hallettmiket/preffect" \
    --with "torch==2.6.*" --index https://download.pytorch.org/whl/cpu
```

Two reasons. The default download of PyTorch carries several gigabytes of GPU
libraries a machine without a GPU cannot use; the CPU build is a few hundred
megabytes. And from release 2.7 on, PyTorch's CPU kernels are slower, dropout
most of all, by about a quarter of a training step on this model; 2.6 is the
package's floor, so nothing is lost by staying on it. The measurements are on
the [speed page](speed.md). With `pip`, install torch first and PREFFECT
second: `pip install "torch==2.6.*" --index-url https://download.pytorch.org/whl/cpu`.

### A copy of the source, if you want to change it

```bash
git clone https://github.com/hallettmiket/preffect.git
cd preffect
bash scripts/install.sh
```

That makes an isolated environment in `.venv`, installs PREFFECT and everything
optional into it, and then checks its own work: that the package imports, that
the `preffect` command runs, and that the fast tests pass. If any of those
fail it says so rather than reporting success. Add `--minimal` for the models,
training and inference alone.

It uses [uv](https://docs.astral.sh/uv/) when uv is installed, because uv is
several times faster, and falls back to the standard library's `venv` and `pip`
when it is not. It will not download and run an installer from the internet to
get uv.

This route leaves the command at `.venv/bin/preffect` rather than on your path,
which is deliberate: the copy you are editing cannot then be confused with one
installed somewhere else.

### Installing it yourself instead

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

That gives you the models, training and inference. Three extras add the rest:

```bash
.venv/bin/pip install -e '.[graph]'    # the single and full models
.venv/bin/pip install -e '.[cluster]'  # grouping samples, scoring the grouping
.venv/bin/pip install -e '.[viz]'      # the above, plus figures
```

### Check it worked

```bash
preffect --version
```

If that prints something unexpected, run it as `python -P -m preffect
--version`. Python puts the current folder at the front of the import path, so
a folder named `preffect` sitting next to where you are standing wins over the
one you installed. `-P` turns that off. This is not hypothetical: it is how the
install script's own check first failed.

**The rest of this page writes the command as `preffect`.** If you took a copy
of the source instead, write `.venv/bin/preffect` everywhere it appears.

## What your data has to look like

One folder per split, each containing an `.h5ad` file, which is the
[AnnData](https://anndata.readthedocs.io/) format:

```
my_study/
    train/
        counts.h5ad
    validation/        # optional
        counts.h5ad
```

Inside that file:

- **`X`** is the counts: rows are samples, columns are genes. **Whole numbers.**
  Not normalised, not log-transformed. PREFFECT models counts, so a
  log-transformed matrix will not fit and the check below will say so.
- **`var_names`** are the gene names, and they have to be unique. They are the
  key everything matches on.
- **`obs`** may carry anything you want to use later: a batch, a subtype.
- **`obsm['sample_sample_adj']`** is the sample-sample network, if you have
  one. Only the `single` and `full` models use it.

## Always run the check first

```bash
preffect check --train my_study/train --validation my_study/validation
```

This reads the data, reports what it found, and refuses if something would stop
a run. It takes seconds and it is the single most useful command here, because
the problems it finds are ones that otherwise show up as a confusing result
rather than as an error.

On real data it has caught, among other things, twenty-one genes all sharing
one name, a validation set that was a byte-for-byte copy of the training set,
and a sample-sample network with no edges in it.

Output on a healthy dataset:

```
training: 1 tissue(s) from my_study/train
  tissue 1: 528 samples, 18790 genes, a network with 105 edge(s), 1.0 per sample
  the counts look usable
validation: 61 samples from my_study/validation
  tissue 1: 0 of 528 training samples reappear in validation
```

## Fitting a model

```bash
preffect train \
    --train my_study/train \
    --validation my_study/validation \
    --out model.pt \
    --epochs 100 \
    --seed 0 \
    --threads 16
```

Two of those are worth knowing about.

**`--seed`** makes the run repeatable. Without it the same command gives a
different model each time.

**`--threads`** matters more than you would expect. PyTorch otherwise starts
one compute thread per core, which on a large shared machine can be over a
hundred, and coordinating them costs more than the arithmetic. The right number
depends on the size of the data. Measured on 528 samples by 18,790 genes,
seconds per epoch:

| threads | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| seconds per epoch | 3.9 | 2.1 | 1.3 | **1.2** | 1.6 |

On small data the ordering reverses and one thread wins. The default of 4 suits
a small test rather than a real cohort.

## Applying it

```bash
preffect apply \
    --model model.pt \
    --data my_study/train \
    --out cleaned.h5ad
```

The result carries the cleaned counts in `X`, the predicted proportions in
`layers['composition']`, and the latent summary in `obsm['latent']`.

Genes are matched to the ones the model was fitted on **by name**, so the
column order of the file you apply to does not have to match the one you
trained on. If a gene is missing, it says so rather than quietly shifting
everything along by one.

## Next

[The command line](command-line.md) is every option of the three commands
above, with what each is for and when to change it. [Using it](guide.md)
covers the parts the command line does not reach: imputation with a held-out
set, clustering, and figures.
