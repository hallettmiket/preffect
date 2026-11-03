# Using it

The command line covers fitting and applying. This page covers the rest, from
Python.

## Fitting from Python

```python
from pathlib import Path

import torch

from preffect.config import (
    DataConfig, LossWeights, ModelConfig, PreffectConfig,
    RuntimeConfig, TrainConfig,
)
from preffect.data.checks import check_tissues
from preffect.data.loading import load_tissues
from preffect.data.tensors import to_tensors
from preffect.model import SimpleModel
from preffect.training import configure_runtime, train

tissues = load_tissues(Path("my_study/train"), n_tissues=1)
check_tissues(tissues, split="training")      # raises if something is wrong
data = to_tensors(tissues[0], correction_variables=())

config = PreffectConfig(
    data=DataConfig(train_path=Path("my_study/train")),
    model=ModelConfig(latent_dim=20, hidden_dim=256),
    loss=LossWeights(),
    train=TrainConfig(epochs=100, batch_size=64, learning_rate=1e-3),
    runtime=RuntimeConfig(seed=0, num_threads=16),
)

configure_runtime(config)                      # before building the model
model = SimpleModel(tissues[0].n_vars, config.model)
history = train(model, data, config)
```

`configure_runtime` sets the seed and the thread count, and it has to run
**before** the model is built. Weights are initialised during construction, so
seeding afterwards leaves the initialisation unseeded and the run
unrepeatable. `train` calls it too, so a run that only uses `train` is covered.

## The full model, and patients who are missing a tissue

The full model reads several profiles from the same patient and predicts the
first one. It needs to know which row of one file is which row of another, and
the files deliberately do not share sample names, so the correspondence is
written down in a table under `obsm['sample_association']` with one column per
tissue. `read_split` finds it, reorders every tissue by it, and hands back
tissues whose row *i* is the same patient throughout.

**A patient can be missing a tissue, and is kept rather than refused.** A
cohort where some patients have no matched normal is the ordinary case. Their
cell in the table is empty, and PREFFECT gives that tissue a row of its own for
them, called a padded row here:

- its counts are the **rounded per-gene mean of that tissue's real samples**;
- every annotation is missing;
- it has **no links** in that tissue's sample-sample network, so it attends to
  nothing and nothing attends to it.

**What a padded row contributes to, and what it does not.** It is left out of
the expression reconstruction term and both library-size terms, which are the
three terms that read the counts. It is kept in the latent and network terms,
so it is encoded and it takes part in the attention. That split is inherited
from version 0.1 rather than designed here; the decision of 2026-11-02 on
unmatched samples measures it against the alternative of dropping those
patients.

Two consequences worth knowing before you fit:

- **Nothing is ever hidden in a padded row.** Its counts are a mean this
  package invented, so recovering one would measure nothing. A mask that hides
  an entry in a padded row is refused by name rather than quietly worked
  around.
- **Padding and correction variables cannot be combined.** A padded row has no
  annotations, so it has no batch and no stage, and a correction variable with
  a missing value is refused. Fill the annotations in yourself, drop the
  incomplete patients, or leave the correction variables out.

**What comes out leaves them behind.** `infer` and `preffect apply` return
results for the real rows of the predicted tissue only, and report how many
padded rows they dropped, in `n_padded_rows_dropped` and in
`uns["padded_rows_dropped"]` of the file `apply` writes.

**`preffect check` tells you how many there are** before you fit anything, one
line per tissue, which is the share of the cohort that will be made up rather
than measured. [The command line page](command-line.md) shows the output.

To drop the incomplete patients instead, ask the alignment to refuse them and
act on what it names:

```python
from preffect.data.tissues import align_tissues, unmatched_counts

print(unmatched_counts(tissues))        # {'tau_1': 0, 'tau_2': 24}
alignment = align_tissues(tissues, unmatched="refuse")   # raises if any are missing
```

## Imputation, measured honestly

To find out how well it fills gaps, hide some entries, fit on what is left, and
score only the hidden ones.

```python
import numpy as np

from preffect.data.masking import apply_mask, mcar_mask
from preffect.data.tensors import TensorData
from preffect.inference import imputation_error, infer

truth = torch.from_numpy(counts)
mask = mcar_mask(n_samples, n_genes, fraction=0.2, rng=np.random.default_rng(0))
shown = torch.from_numpy(apply_mask(counts, mask))

model_data = TensorData(inputs=shown, targets=truth, mask=mask)
train(model, model_data, config)

result = infer(model, model_data)
scores = imputation_error(
    result.expected_counts, truth,
    hidden=torch.from_numpy(mask.hidden), kind="mean_relative",
)
print(scores["hidden"], scores["visible"])
```

To stop training where that error settles rather than where the loss does,
which is ten times sooner, ask for it in the settings and hide entries as
above: `TrainConfig(early_stopping_patience=1000, stop_on="imputation")`. The
[command line page](command-line.md) says what the two rules measure and
when each is the right one.

!!! warning "Truth is an argument, with no default"
    `imputation_error` will not invent something to compare against. Version
    0.1 looked for a layer called `truth`, and when it was absent the data
    loader supplied the per-gene mean, so on real data it reported the distance
    from the gene average and called it imputation accuracy. There is nowhere
    for a fabricated truth to come from here.

**The library size is corrected for what the hiding took away.** A sample's
library size is its total count, and the model multiplies everything it
predicts for that sample by it. Read off the counts the model was shown, that
total is short by the hidden fraction, while the targets keep the originals,
so every predicted count would be short by the same fraction and nothing in
the model could make it up: the composition sums to one across genes, so the
library alone sets the scale. Passing a `mask` to `TensorData`, as above, is
all it takes. The correction is the number of genes divided by the number of
that sample's entries the model was shown, and it travels with the data
through training, mini-batching and inference.

The correction reads **how many** entries were hidden and never which values.
A sample's true total together with its visible entries would give the sum of
its hidden ones exactly, so a library taken from the targets would hand the
model the answer it is being scored on. It is also available for missing
completely at random only, which is the one scheme `mcar_mask` writes; a
scheme that hid entries according to their values would make the estimate
biased rather than unavailable, and is refused by name.

Scoring the visible entries too is worth doing. Doing well on entries the model
was shown is not imputation, and if the two numbers are similar the model may
be reproducing its input rather than generalising.

**Relative error leaves out entries whose true value is zero**, and reports how
many it left out. Dividing by zero makes an average meaningless, and counts are
10 to 55 per cent zeros.

## Scoring a predicted network

The single and full models reconstruct the sample-sample network from a
latent space of their own, and the loss cannot say whether they are getting
it right: on a network one per cent dense, a model that predicts no edge
anywhere scores well on an unweighted cross-entropy (issue #13). Score the
edges instead.

```python
from preffect.inference import network_recovery

model.eval()
with torch.no_grad():
    output = model(data.inputs, data.corrections, data.edge_index)
    predicted = model.network_decoder(output.network_latent_mean, data.corrections)

scores = network_recovery(predicted, data.adjacency_matrix())
print(scores)
# precision 0.021, recall 1.000, F1 0.041 at one half (111259 pairs predicted,
# 2355 edges among 179700 pairs); average precision 0.039 against 0.013 for a
# flat prediction
```

That output is a single model after 200 epochs on `exp/17`'s synthetic cohort
with the term balanced: the ranking is three times better than chance, and
the threshold of one half calls far too many pairs, which is the kind of thing
the loss alone would never have shown.

Both inputs are square over the same samples and are read as symmetric: each
pair is counted once and the diagonal is ignored. An edge is predicted where
the probability exceeds one half. Average precision ranks every pair by its
probability and needs no threshold; a model that cannot rank the pairs at
all, whether it predicts no edge, every edge, or one half everywhere, scores
the density there, which the result carries as `flat_average_precision` so
the baseline is beside the number. Predicting no edge anywhere scores zero on
precision, recall and F1.

Decode from the network latent **mean**, as above, rather than from a draw,
for the same reason inference uses the latent mean: the same model over the
same data should give the same network twice. For the full model the decoder
and the latent are lists, one per tissue.

## Clustering

```python
from preffect.clustering import clustering_recovery, resolution_for_target

grouping = resolution_for_target(result.latent.numpy(), target_clusters=5, seed=0)
print(grouping.n_clusters, "clusters at resolution", grouping.resolution)

scores = clustering_recovery(grouping, subtypes)   # subtypes from adata.obs
print(scores)
# adjusted Rand 0.412, adjusted mutual information 0.389
# (5 clusters against 5 known groups, 528 samples)
```

Both scores are corrected for chance. That matters at these sizes: with five
groups and a few hundred samples, an uncorrected Rand index sits near 0.7 for a
random grouping and reads as success.

`target_clusters` is a **ceiling**, not a promise. The search only lowers the
resolution, so it stops as soon as the count is low enough and can return
fewer. If it runs out of room, `grouping.hit_the_floor` is true and
`grouping.resolution` is the resolution the labels actually came from.

## Figures

Every figure returns a matplotlib `Figure` and none of them write a file.

```python
from preffect import plots

figure = plots.library_size_and_dispersion(result, counts)
plots.save_figure(figure, "figures/diagnostics.png")

plots.save_figure(
    plots.subtype_heatmap(counts, gene_names, subtypes, normalise="log_z_score"),
    "figures/pam50.png",
)
```

Available: `library_size_histogram`, `gene_scatter`, `batch_adjustment`,
`zero_fraction_against_pi`, `library_size_and_dispersion`, `latent_umap`,
`loss_curves`, `subtype_heatmap`.

Two things to know:

- **The UMAP is seeded.** Version 0.1's was not, so the same model and data drew
  a different picture every run. Seeding costs parallelism, which UMAP will warn
  about; a figure you cannot reproduce is not worth the time it saved.
- **`subtype_heatmap` finds the PAM50 genes whichever way your data names
  them**, symbols or Ensembl identifiers, and raises if it finds too few rather
  than drawing an empty figure.

## Choosing genes

```python
from preffect.data.selection import apply_gene_selection, choose_genes
from preffect.gene_sets import PAM50

selection = choose_genes(tissues[0], n_genes=2000, force_genes=PAM50)
tissues = apply_gene_selection(tissues, selection, "training")
```

Genes are ranked by interquartile range, most variable first. `force_genes`
keeps a panel regardless of its variability.

!!! warning "A panel that matches nothing raises"
    If none of the panel is in the data, the naming is wrong, and forcing it
    would silently do nothing. Version 0.1 matched an Ensembl list against
    symbol-named data in three of its own PAM50 notebooks and forced zero genes
    without a word. Passing a `GeneSet` rather than a list of names avoids the
    problem entirely, because both namings are tried.

**Choose genes once, from the training data, and apply the same selection to
every split by name.** Re-choosing per split gives different genes for the same
column, which is a leak and a silent one.

## When things go wrong

Everything raises `PreffectError` or one of its subclasses, and the message
names the setting, what was asked for, and what to do instead.

```python
from preffect.errors import ConfigError, DataError, PreffectError
```

If a message tells you only that something is wrong and not what to do about
it, that is a defect in the message. Version 0.2 has had three of those and
they are all in the findings list.
