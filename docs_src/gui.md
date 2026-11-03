# The graphical interface

A page in your browser for the part of the work you want to look at: whether a
folder of counts is usable, what a fitted model makes of it, and the figures
that say whether to believe it.

```bash
preffect gui
```

That starts a local server and opens the page. It needs the `gui` extra:
`pip install "preffect[gui]"`, or `uv tool install "preffect[gui] @ git+https://github.com/hallettmiket/preffect"`. `preffect gui` says so if it is
missing.

**It answers on this machine only and sends nothing anywhere.** The page is
served to `localhost` and to nothing else, so nobody on the network can open
it, and Streamlit's own usage reporting, which the library it is built on
turns on by default, is turned off by the command that starts it. Press
Ctrl+C in the terminal to stop it. `--port` changes the port if 8501 is taken.

If PREFFECT runs on a server and you are at a laptop, forward the port over
SSH and open the same address in your own browser:

```bash
ssh -L 8501:localhost:8501 your-server    # then preffect gui on the server
```

## What it does

**Check.** Give it a folder of counts and it reports what `preffect check`
reports: samples and genes per tissue, whether there is a network and how many
edges it has, and, if something would stop a run, the sentence the run would
have refused with.

**Apply.** Give it a saved model as well, press the button, and it runs the
model over the counts and shows:

- what kind of model it was, which count distribution it was fitted with, and
  how many samples and genes were matched
- a button to download the cleaned counts as an `.h5ad` file, the same file
  `preffect apply` writes
- the before-and-after figures: observed library sizes against inferred,
  observed against fitted counts for the most variable genes, gene means and
  dispersion, and, for a zero-inflated model, how often each gene is zero
  against the model's own zero probability
- the latent summary of the samples in two dimensions, coloured by any column
  of the sample table with twenty or fewer values, such as a subtype or a batch

Every figure is one of [`preffect.plots`](reference/analysis.md)' figures,
drawn by the same call a script would make, so anything seen here can be
reproduced and put in a paper.

## What it does not do

**It does not fit a model.** Training a real cohort is hours, and belongs on
the command line where it can be left running and logged. The interface reads
what `preffect train` wrote. That is why the count distribution is shown here
and chosen there, with
[`preffect train --likelihood`](command-line.md): a zero-inflated model and a
plain one of the same kind look alike on this page and predict different
counts, so the page owes you the answer even though it cannot change it.

**It does not score imputation or clustering.** Both need a held-out set or
known labels and a decision about what to compare; [Using it](guide.md) covers
them from Python.

## If something is wrong

A folder that cannot be read, counts that would stop a run, or a device the
machine lacks shows as a sentence on the page, the same sentence the command
line prints. If the latent figure says it needs the `viz` extra, the `gui`
extra includes it, so that is an environment assembled by hand; install it.
