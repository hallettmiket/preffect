# PREFFECT 0.2

A statistical model for RNA sequencing counts from formalin-fixed
paraffin-embedded tissue, usually written FFPE.

FFPE is how most clinical tumour samples are stored, and it damages RNA. Counts
from FFPE material are noisier than counts from fresh-frozen tissue, and some
values are missing outright. PREFFECT fits a model of what the counts would
have looked like without that damage, and fills the gaps.

## What it does

It learns a short summary of each sample, a few dozen numbers rather than
twenty thousand, and reconstructs the counts from that summary. Three things
follow:

- **Cleaned counts.** What the model thinks each sample should look like.
- **Filled gaps.** Values for entries that were missing or masked.
- **A latent summary** of each sample, which can be grouped to find subtypes.

Counts are modelled with a negative binomial distribution, which is the usual
choice for sequencing counts because they vary more than a Poisson allows. A
zero-inflated version is available for data with more zeros than that explains.

## Three models

| Model | What it uses | When |
|---|---|---|
| `simple` | counts alone | the default, and the right starting point |
| `single` | counts plus one sample-sample network | when you know which samples resemble which |
| `full` | several matched tissues per patient | when you have them |

Start with `simple`. The graph models cost roughly ten times the parameters and
twenty times the time, and they only pay for that if the network has real edges
in it. [None of the datasets this was built against did.](changes.md)

## Where to go

- **[Getting started](getting-started.md)** installs it and fits a model.
- **[The command line](command-line.md)** is every command and every option,
  with what each is for and when to change it.
- **[The graphical interface](gui.md)** checks a folder, applies a saved model
  and shows the figures, in a browser, on your own machine.
- **[Using it](guide.md)** covers the real workflow: checking data, choosing
  genes, imputing, clustering, drawing.
- **[Reference](reference/config.md)** is the API, taken from the docstrings.
- **[What changed from 0.1](changes.md)** is what moves a number, and why.

## A note on this version

Version 0.2 is a rewrite, not a patch. The [list of fixes relative to version
0.1](https://github.com/hallettmiket/preffect/blob/main/CHANGELOG.md)
is kept in the repository because most of the design follows from it. Several
of those defects were silent: a model kind quietly swapped for another, a term
computed and then left out of the sum, an experiment that compared a model with
itself.

The rule the rewrite is built around is that **nothing is silently rewritten**.
If a setting cannot be honoured, it raises and says which setting, what was
asked for, and what to do instead. That is less convenient and much easier to
trust.
