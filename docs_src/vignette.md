# A worked example

Fit a model, check it, and apply it to data it has never seen. About two
minutes end to end.

## The data

```bash
.venv/bin/preffect example-data --out vignette_data
```

```
  simple:
    train       120 samples  md5 5130e40e
    validation   40 samples  md5 9df2e527
    test         40 samples  md5 0e8dbb61
```

Three splits, **no sample in any two of them**, and three different files. The
command checks both before it reports success and refuses to write otherwise.

!!! warning "Why that is worth checking"
    Version 0.1's vignettes ship a `test/` folder byte-identical to
    `validation/`, in all three of simple, single and full. Published Figure S4
    shows a user pointing inference at that folder while the caption calls it a
    held-out test set. It is the validation set. Issue #11.

The counts are simulated from a negative binomial, which is what PREFFECT
models, with three groups of samples that each raise a different block of
genes. So there is something real to find, and you can tell whether the model
found it.

## 1. Check the data first

```bash
.venv/bin/preffect check --train vignette_data/simple/train \
                         --validation vignette_data/simple/validation
```

```
training: 1 tissue(s) from vignette_data/simple/train
  tissue 1: 120 samples, 200 genes, no network
  the counts look usable
validation: 40 samples from vignette_data/simple/validation
  tissue 1: 0 of 120 training samples reappear in validation
```

That last line is the one to read. On your own data it tells you whether your
splits are what you think they are, and it is how the vignette problem above
would have been caught.

## 2. Fit

```bash
.venv/bin/preffect train --train vignette_data/simple/train \
                         --validation vignette_data/simple/validation \
                         --out model.pt \
                         --epochs 80 --batch-size 40 --seed 0 --threads 4 \
                         --patience 5
```

```
fitting a simple model: 200 genes, 36,090 parameters, 80 epochs
finished after 8 epoch(s)
  training loss   1456.05 -> 947.48
  validation loss 1119.80
  stopped early, best epoch was 3
wrote model.pt
```

Validation close to training means it is not simply memorising. A validation
loss far above the training loss means it is, and a toy this small gets there
within a few epochs: 120 samples are memorised long before 80 epochs are up,
and without `--patience` the validation loss climbs from epoch 3 to the end of
the run while the training loss keeps falling. `--patience 5` stops the run
once the validation loss has not fallen for five epochs in a row, and the
model written is the best epoch's, not the last. A real cohort takes hundreds
of epochs to settle, and the loss settling is not the whole story: see
`--patience` on [the command line page](command-line.md).

**Always pass `--seed`** if you want the same model twice, and **set
`--threads`** — the right number depends on the size of your data, and the
default of 4 suits a small example rather than a real cohort. See
[getting started](getting-started.md).

## 3. Apply it to data it has never seen

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

## If your cohort is too small to divide

Leave `--validation` out. PREFFECT says so rather than inventing one:

```
validation: none given, so no validation loss will be reported
```

That is honest. Splitting 40 samples three ways gives you nothing you can
trust, and copying one split into another, which is what version 0.1 did,
gives you a number that looks like a held-out result and is not.

## The graph model

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
