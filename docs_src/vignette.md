# A worked example

Fit a model, check it, apply it to data it has never seen, and score what it
filled in. About three minutes end to end.

## The data

This step writes a small simulated cohort to work on, and says what it wrote.

```bash
.venv/bin/preffect example-data --out vignette_data
```

```
wrote vignette_data
  simple:
    train       120 samples  md5 4091e1ae
    validation   40 samples  md5 51f7b50a
    test         40 samples  md5 073eb2ee
  single:
    train       120 samples  md5 8ae37503
    validation   40 samples  md5 69d0f1e8
    test         40 samples  md5 2778a063
  full:
    train       120 samples  md5 fed1a84d  2 tissues
    validation   40 samples  md5 d0f02ad3  2 tissues
    test         40 samples  md5 42a999d2  2 tissues

200 genes, 3 groups, negative binomial counts.
No sample appears in more than one split, and the three files differ.
The genes are drawn once per cohort, so every split is the same genes.
Batch B has 40 of the 200 genes raised 1.3 times, for the correction variables to take out.
Every file carries the count each entry was drawn from in layers['true_expected_counts'].
The full model's two tissues are the same patients under different names.
```

Three splits, **no sample in any two of them**, and three different files. The
command checks both before it reports success and refuses to write otherwise.

The checksums above are from the Python 3.12 library stack. The counts, the
names and the batch labels are the same on Python 3.11, which resolves an
older anndata and an older pandas, but the bytes those libraries write around
them are not, so the checksums there differ. Compare them against another run
of your own install, not against this page.

!!! warning "Why that is worth checking"
    Version 0.1's vignettes ship a `test/` folder byte-identical to
    `validation/`, in all three of simple, single and full. Published Figure S4
    shows a user pointing inference at that folder while the caption calls it a
    held-out test set. It is the validation set. Issue #11.

The counts are simulated from a negative binomial, which is what PREFFECT
models, with three groups of samples that each raise a different block of
genes. So there is something real to find, and you can tell whether the model
found it.

**The three splits are the same genes.** Each gene has one level and one
amount of noise, drawn once for the whole cohort, and every split is drawn
from those. Without that a model fitted on the training split has nothing to
say about the test split, however well it fits: until finding 65 this
generator drew the genes separately for each split, so the per-gene means of
the training and test splits were uncorrelated and most of this page's
numbers were measuring that rather than measuring PREFFECT.

**Batch B's own genes are raised**, by a factor of 1.3, in a fifth of the
genes picked at random and listed in the file under
`uns['batch_shifted_genes']`. It is deliberately smaller than the four-fold
lift the groups give their blocks, so the groups still dominate and the batch
is the second thing you see. What it is there for is the **correction
variables**, which take a known nuisance label out of what the encoder is
given and are asked for from Python rather than from the command line
([using it](guide.md) shows where), and the batch figure in
[the interface](gui.md). It is not a
demonstration of the objective's batch-centroid term, which no run here
reaches: training passes no batch labels to the loss, so that term never
fires.

**Every file also carries the count each entry was drawn from**, in
`layers['true_expected_counts']`. That is the negative binomial mean, before
the draw added its noise. No real cohort can have it; it is here so that step
4 can score a filled-in entry against what the entry should have been rather
than against one noisy draw of it.

## 1. Check the data first

This step reads the files and says what is in them, before anything is fitted.

```bash
.venv/bin/preffect check --train vignette_data/simple/train \
                         --validation vignette_data/simple/validation
```

```
training: 1 tissue(s) from vignette_data/simple/train
  tissue 1: 120 samples, 200 genes, no network
  tissue 1: carries the count each entry was drawn from in layers['true_expected_counts'], so these counts are simulated
  the counts look usable
validation: 40 samples from vignette_data/simple/validation
  tissue 1: 0 of 120 training samples reappear in validation
```

The last line is the one to read. On your own data it tells you whether your
splits are what you think they are, and it is how the vignette problem above
would have been caught. The line above it will not appear on your own data:
only something simulated can carry the count each entry was drawn from, so
finding it is how `check` tells you the file was generated rather than
measured.

## 2. Fit

This step fits the model and stops it where it stops improving.

```bash
.venv/bin/preffect train --train vignette_data/simple/train \
                         --validation vignette_data/simple/validation \
                         --out model.pt \
                         --epochs 80 --batch-size 40 --seed 0 --threads 4 \
                         --patience 5
```

```
fitting a simple model: 200 genes, 36,090 parameters, 80 epochs
finished after 67 epoch(s)
  training loss   1218.02 -> 828.70
  validation loss 804.63
  stopped early, best epoch was 62
wrote model.pt
```

Validation close to training means it is not simply memorising, and here the
two end within three per cent of each other. `--patience 5` stops the run once
the validation loss has not fallen for five epochs in a row, and the model
written is the best epoch's, not the last. Here it fires at epoch 67 of a
budget of 80 and keeps epoch 62, which is where more fitting stopped buying
anything. A real cohort takes hundreds of epochs to settle, and the loss
settling is not the whole story: see `--patience` on
[the command line page](command-line.md).

!!! note "Where 80 epochs comes from"
    Fitted at four lengths on this data, on four threads of a CPU. The column
    headed *test fit* is the median across genes of the correlation between
    the model's expected counts and the actual counts of the 40 test samples,
    which is what this example exists to produce. With `--patience 5` the runs
    budgeted 200 and 500 epochs both stop at 67 and give the 80-epoch row.

    | Budget | Epochs run | Training loss | Validation loss | Test fit | Verdict |
    |---|---|---|---|---|---|
    | 20 | 20 | 890.14 | 868.74 | 0.473 | too short: the test counts are not reproduced |
    | 80 | 67, stopped early | 828.70 | 804.63 | 0.784 | the shortest that reproduces them |
    | 200 | 200 | 815.59 | 801.20 | 0.787 | no better on the test split, three times the work |
    | 500 | 500 | 800.43 | 803.60 | 0.784 | worse: the training loss keeps falling, the validation loss does not |

**Always pass `--seed`** if you want the same model twice, and **set
`--threads`**. The right number depends on the size of your data, and the
default of 4 suits a small example rather than a real cohort. See
[getting started](getting-started.md).

## 3. Apply it to data it has never seen

This step runs the fitted model over the test split and checks what came out.

```bash
.venv/bin/preffect apply --model model.pt \
                         --data vignette_data/simple/test \
                         --out cleaned.h5ad --threads 4
```

```
loaded a simple model written by PREFFECT 0.2.0, applying on cpu
matched 200 genes to the ones the model was fitted on
wrote cleaned.h5ad: 40 samples, 200 genes
```

`cleaned.h5ad` carries the adjusted counts in `X`, the predicted proportions in
`layers['composition']`, and the latent summary in `obsm['latent']`.

**This is the step the published figure got wrong.** The test split was never
seen during fitting and never used to decide when to stop, so what comes out is
a genuine held-out result.

Two things are worth measuring on it. The first is whether the expected counts
track the real ones: across the 40 test samples the median gene's correlation
between the two is 0.78, and 162 of the 200 genes are above 0.5. The second is
whether the latent summary has found the three groups, which the package will
score for you. Clustering needs extra packages: `pip install
'preffect[cluster]'`.

```python
import anndata as ad

from preffect.clustering import clustering_recovery, resolution_for_target

cleaned = ad.read_h5ad("cleaned.h5ad")
test = ad.read_h5ad("vignette_data/simple/test/counts.tau_1.h5ad")

found = resolution_for_target(cleaned.obsm["latent"], target_clusters=3, seed=0)
print(clustering_recovery(found.labels, test.obs["subtype"].astype(str)))
# adjusted Rand 1.000, adjusted mutual information 1.000
# (3 clusters against 3 known groups, 40 samples)
```

Both scores are corrected for chance, so 1.000 means the clustering of the
latent space is the known grouping exactly, on samples the model never saw.

## 4. Score what it filled in

This step hides a fifth of the training entries, fits again on what is left,
and scores the hidden entries against the counts they were drawn from.

```bash
.venv/bin/preffect train --train vignette_data/simple/train \
                         --validation vignette_data/simple/validation \
                         --out imputing.pt --hide 0.2 \
                         --epochs 80 --batch-size 40 --seed 0 --threads 4 \
                         --patience 5
```

```
hiding 20% of the training entries; the model is scored against the full counts
fitting a simple model: 200 genes, 36,090 parameters, 80 epochs
finished after 80 epoch(s)
  training loss   1160.82 -> 826.56
  validation loss 805.03
  imputation error on hidden entries 0.646, over 4,747 entries (53 hidden entries with a true value of zero left out)
wrote imputing.pt
```

That 0.646 is the average relative distance from the **count that was
observed**, and most of it is noise rather than error: a count is one draw
from a negative binomial, and nothing can predict the draw. Scoring against
the mean it was drawn from instead separates the two, and on simulated data
that mean is in the file.

```python
from pathlib import Path

import numpy as np
import torch

from preffect.checkpoint import load_checkpoint
from preffect.inference import imputation_error, infer
from preffect.workflow import build_model, hide_entries, read_split, to_model_data

checkpoint = load_checkpoint("imputing.pt")
model = build_model(
    checkpoint.model_config.kind, checkpoint.n_genes, checkpoint.model_config, 1
)
model.load_state_dict(checkpoint.model_state)
model.eval()

# The same entries the run hid: the fraction and the seed are what choose them.
tissues = read_split(Path("vignette_data/simple/train"), 1)
data = hide_entries(to_model_data(tissues, checkpoint.model_config, False), 0.2, 0)
truth = torch.from_numpy(
    np.asarray(tissues[0].layers["true_expected_counts"], dtype=np.float32)
)

scores = imputation_error(
    infer(model, data).expected_counts, truth,
    hidden=torch.from_numpy(data.mask.hidden), kind="mean_relative",
)
print(round(scores["hidden"], 4), round(scores["visible"], 4))
# 0.1232 0.1279
```

So the model lands within about 12 per cent of what each hidden entry should
have been, against 65 per cent of the single count that was actually drawn.
The two numbers are the same measure on different comparisons, and the gap
between them is the negative binomial noise. Nothing is skipped here: a true
expected count is never zero, while 53 of the observed counts at hidden
entries were, and a relative error cannot be computed against zero.

**Your own data has no such layer, and PREFFECT will not invent one.** Version
0.1 looked for a layer called `truth`, and where it was absent the loader
supplied the per-gene mean, so on real data it reported the distance from the
gene average and called it imputation accuracy. On a real cohort the honest
comparison is the observed count, which is the 0.646 above. [Using
it](guide.md) says more about what that number does and does not measure.

## If your cohort is too small to divide

This section is for a cohort that cannot give up samples to a validation set.

Leave `--validation` out. PREFFECT says so rather than inventing one:

```
validation: none given, so no validation loss will be reported
```

That is honest. Splitting 40 samples three ways gives you nothing you can
trust, and copying one split into another, which is what version 0.1 did,
gives you a number that looks like a held-out result and is not.

## The graph model

This section fits the model that reads a sample-sample network as well as the
counts.

`vignette_data/single/` carries the same thing plus a sample-sample network
with real edges in it.

```bash
.venv/bin/preffect check --train vignette_data/single/train
#   tissue 1: 120 samples, 200 genes, a network with 240 edge(s), 4.0 per sample
```

```bash
.venv/bin/preffect train --train vignette_data/single/train --kind single \
                         --out single.pt --epochs 40 --batch-size 40 \
                         --seed 0 --threads 4
```

```
fitting a single model: 200 genes, 397,530 parameters, 40 epochs
finished after 40 epoch(s)
  training loss   526.47 -> 428.39
wrote single.pt
```

!!! note "Check your network has edges before fitting a graph model"
    `check` reports the edge count because every real dataset this was built
    against carried an identity matrix there, which is a network with no edges
    at all. A graph model on one of those costs about ten times the
    parameters of the simple model and has nothing to attend to. It now refuses
    rather than training on nothing. Findings 35 and 54.

## Next

- The same steps as a notebook: `vignettes/preffect_vignette.ipynb` in the
  repository.
- [Using it](guide.md) for imputation, clustering and figures from Python.
- [Reference](reference/config.md) for the settings.
