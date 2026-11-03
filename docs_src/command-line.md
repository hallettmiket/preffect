# The command line

Six commands, in the order you meet them. The first two turn what your study
already has into the files PREFFECT reads, and are needed once; the next three
are the work itself. A seventh opens the same work [in a browser](gui.md):

| Command | What it does |
|---|---|
| `preffect prepare` | builds the tissue files from a counts table and a sample table |
| `preffect network` | adds a sample-sample network to tissue files that have none |
| `preffect check` | reads the counts and says whether they can be used, without fitting anything |
| `preffect train` | fits a model and saves it to one file |
| `preffect apply` | runs a saved model over counts and writes the cleaned result |
| `preffect example-data` | writes a small simulated cohort to try the others on |

`preffect --help` lists the commands, `preffect <command> --help` lists a
command's options with their defaults, and `preffect --version` prints the
version. This page says the same things at more length: what each option is
for, and when to change it.

The examples write `preffect`. If you installed a copy of the source with
`scripts/install.sh`, write `.venv/bin/preffect`. If `preffect` is not on your
path at all, or prints something you did not expect, `python -P -m preffect`
runs the same thing. `-P` stops Python putting the current folder first on the
import path, so a folder named `preffect` beside you cannot be mistaken for the
installed package.

## What every command shares

**A data folder, not a file.** Each of `--train`, `--validation` and `--data`
names a folder holding one file per tissue, named `<anything>tau_<n>.h5ad` with
`<n>` counting from 1. A single-tissue cohort, which is the usual case, has one
file ending `tau_1.h5ad`. [Getting started](getting-started.md) says what has
to be inside it.

**`--tissues` is how many of those files to expect.** A tissue here is one
sample from each patient under one condition, such as tumour and adjacent
normal; the `full` model fits several at once. The default of 1 is right for
everything except the `full` model.

**It says what it is doing.** Every command prints what it read and what it
produced, so a long run that says nothing can be told apart from one that has
hung, and a run that produced nothing cannot look like one that succeeded.

**It stops rather than guesses.** A setting that cannot be honoured, a folder
that does not exist, or an output path that cannot be written stops the
command with one sentence naming the problem, before any work is done. The
exit status is 0 when the command did what was asked, 2 when it refused, and 1
when no command was given. A refusal prints no traceback, because a traceback
tells someone who did not write the code nothing they can act on.

## `preffect prepare`

Turns a table of counts and a table describing the samples into the files the
other commands read. Run it once, when your study's counts are still in
whatever a sequencing pipeline wrote.

```bash
preffect prepare --counts counts.csv --samples samples.csv --out my_study/train
```

| Option | What it does | Default |
|---|---|---|
| `--counts` | a comma- or tab-separated table of counts; give it once per tissue | required |
| `--samples` | a table describing those samples, once per tissue, in the same order | required |
| `--out` | the folder to write the tissue files in | required |
| `--counts-are` | which way round the counts table is written | `samples-by-genes` |
| `--sample-column` | which column of the sample table holds the sample names | the first |
| `--name` | what the files are called before the tissue number | `counts` |
| `--patient-column` | which column says which patient a sample came from | none |
| `--neighbours` | also build a sample-sample network, with this many links per sample | none |
| `--mutual` | keep a link only where each sample is among the other's nearest | off |
| `--label` | a column of the sample table to measure the network against | none |
| `--overwrite` | write over files already in `--out` | off |

**The counts are read with one row per sample**, the sample names in the first
column and the gene names in the first row. A table written the other way
round, one row per gene, is common; `--counts-are genes-by-samples` turns it
the right way round. The counts have to be whole numbers that are not
negative, and a table that is not says which entries are at fault.

Output on a 60-sample study:

```
tissue 1: 60 samples and 400 genes from counts.csv, read as samples-by-genes
  3 annotation(s) per sample from samples.csv
  tissue 1: 60 samples, 371 edge(s), 12.4 per sample (fewest 10, middle 11, most 19); no isolated samples
  tissue 1: 100.0% of edges join two samples with the same subtype, against 32.2% if the edges ignored it
wrote my_study/train/counts.tau_1.h5ad
check it with: preffect check --train my_study/train
```

**Several tissues at once, for the full model.** Give `--counts` and
`--samples` once per tissue and name the column that says which patient each
sample came from. That column is the only thing linking a sample in one tissue
to a sample in another, since the files do not share sample names, and the
full model cannot be fitted without it. A patient with no sample in some
tissue is kept, with a gap recorded for that tissue, which is the case the
full model pads:

```
tissue 1: 60 samples and 400 genes from counts.csv, read as samples-by-genes
tissue 2: 48 samples and 400 genes from counts_normal.csv, read as samples-by-genes
60 patient(s) across 2 tissues
  tissue 1: every patient has a sample here
  tissue 2: 12 patient(s) have no sample here, and the full model pads a row of per-gene means in for each of them
wrote my_study/full/counts.tau_1.h5ad
wrote my_study/full/counts.tau_2.h5ad
```

## `preffect network`

Builds a sample-sample network from the counts themselves and writes a copy of
each tissue file carrying it. Only the `single` and `full` models use a
network; the `simple` model does not, and does not need this command.

```bash
preffect network --data my_study/train --out my_study/train_linked
```

| Option | What it does | Default |
|---|---|---|
| `--data` | the folder of counts to read | required |
| `--out` | the folder to write the copies in, under the same file names | required |
| `--tissues` | how many tissue files the folder holds | 1 |
| `--neighbours` | link each sample to this many others | 10 |
| `--mutual` | keep a link only where each sample is among the other's nearest | off |
| `--label` | a sample annotation to measure the network against | none |
| `--replace` | build a new network for a file that already carries one with links in it | off |
| `--overwrite` | write over files already in `--out` | off |

**The files it reads are left exactly as they are.** It writes copies, which
is both this project's rule for source data and the only safe thing to do to
somebody's only copy.

```
tissue 1: my_study/train/counts.tau_1.h5ad
  wrote my_study/train_linked/counts.tau_1.h5ad
  60 samples, 371 edge(s), 12.4 per sample (fewest 10, middle 11, most 19); no isolated samples
  100.0% of edges join two samples with the same subtype, against 32.2% if the edges ignored it
the counts are unchanged; fit a graph model on them with preffect train --train my_study/train_linked --kind single
```

Three lines there are worth reading closely.

- **The links per sample is larger than the number asked for**, here 12.4
  against 10, because a link is kept wherever either of the two samples names
  the other. A sample that nothing else names keeps exactly ten.
- **`--label` is how you find out whether the network carries anything.** The
  second line compares the share of links joining two samples with the same
  value of that annotation against the share that would do so if the links
  ignored it. Far above chance means the network found real structure. The
  two being equal, as in `31.2% ... against 31.9%`, means it found none, and a
  graph model over it will cost ten times the parameters of the simple model
  for nothing.
- **A sample with no link at all** is reported where there is one. It happens
  with `--mutual`, and a graph model learns nothing for such a sample.

**Where the recipe comes from.** Each sample is linked to the ten others whose
square-rooted counts correlate with it most. Ten neighbours is the setting
measured on this project's cohort: the one network under which the graph model
beat the simple model, by 8 per cent lower imputation error at hidden entries
in six seeds of six.

## `preffect check`

Reads the data, reports what it found, and refuses if something would stop a
run. It takes seconds and it is the single most useful command here, because
the problems it finds otherwise show up as a confusing result rather than as
an error.

```bash
preffect check --train my_study/train --validation my_study/validation
```

| Option | What it does | Default |
|---|---|---|
| `--train` | folder of training counts | required |
| `--validation` | folder of validation counts | none |
| `--tissues` | how many tissue files each folder holds | 1 |

Output on the vignette data:

```
training: 1 tissue(s) from vignette_data/simple/train
  tissue 1: 120 samples, 200 genes, no network
  tissue 1: carries the count each entry was drawn from in layers['true_expected_counts'], so these counts are simulated
  the counts look usable
validation: 40 samples from vignette_data/simple/validation
  tissue 1: 0 of 120 training samples reappear in validation
```

Four lines are worth reading closely.

- **The network.** `no network` means the file carries no sample-sample
  network, so only the `simple` model applies. `a network with 240 edge(s),
  4.0 per sample` means the graph models have something to use. `a network
  with NO edges between samples` means the file carries a network that links
  every sample to itself and to nothing else. A graph model on one of
  those costs ten times the parameters and has nothing to attend to, so
  `train` refuses it.
- **The overlap.** `0 of 120 training samples reappear in validation` is the
  line that tells you your splits are what you think they are. If every
  training sample reappears, it says `the validation data is the training
  data. Leave --validation out instead.`
- **Without a validation set** it says `validation: none given, so no
  validation loss will be reported`, and that is all it does. It does not
  invent one.
- **Simulated data says so.** The line about
  `layers['true_expected_counts']` appears only where a file carries the count
  each entry was drawn from, which only something generated can. It will not
  appear on your own data. The [worked example](vignette.md) uses that layer
  to score filled-in entries against the signal rather than against one noisy
  draw of it.

With `--tissues 2` or more it adds one line per tissue saying how many
patients have no sample there:

```
training: 2 tissue(s) from my_study/train
  tissue 1: 120 samples, 200 genes, a network with 240 edge(s), 4.0 per sample
  tissue 2: 96 samples, 200 genes, a network with 192 edge(s), 4.0 per sample
  tissue 1: every patient has a sample here
  tissue 2: 24 patient(s) have no sample here, and get a row of per-gene means padded in
  the counts look usable
```

Those 24 patients are kept, and the full model is given a made-up profile for
the tissue they lack: the rounded per-gene mean of that tissue's real samples,
linked to nothing in that tissue's network. The guide's full-model section says
what such a row does and does not contribute to, and the decision of 2026-11-02
on unmatched samples says why it works that way. The number is worth reading,
because it is the share of the cohort the model will be working on that was
made up rather than measured.

## `preffect train`

Fits a model to the training counts and writes it to one file.

```bash
preffect train --train my_study/train --validation my_study/validation \
               --out model.pt --epochs 100 --seed 0 --threads 16
```

**The data**

| Option | What it does | Default |
|---|---|---|
| `--train` | folder of training counts | required |
| `--validation` | folder of validation counts, used to report a loss on samples the model is not fitted to | none |
| `--out` | the file to write the model to; its folder is created if missing, and the path is checked for writability before any work starts | required |
| `--tissues` | how many tissue files each folder holds | 1 |
| `--genes` | keep only this many of the most variable genes, chosen on the training data and applied to the validation data by name | all genes |

**The model**

| Option | What it does | Default |
|---|---|---|
| `--kind` | `simple` uses counts alone; `single` adds one sample-sample network; `full` uses several matched tissues per patient | `simple` |
| `--likelihood` | which count distribution the model fits: `NB`, the negative binomial, or `ZINB`, which adds a separate probability that a count is zero for a reason unrelated to expression level; see below | `NB` |
| `--latent-dim` | the size of the summary the model learns for each sample | 20 |
| `--hidden-dim` | the width of the hidden layers between the counts and that summary | 50 |

**The fitting**

| Option | What it does | Default |
|---|---|---|
| `--epochs` | how many passes over the training data; `--patience` below says how many is enough, and for what | 100 |
| `--batch-size` | how many samples go through the model per step | 100 |
| `--learning-rate` | how far the optimiser moves per step | 0.001 |
| `--patience` | stop early once the loss has not improved for this many epochs; the model kept is the best epoch's | never stop early |
| `--min-delta` | how much the loss must fall to count as improving, for `--patience` | 0 |
| `--stop-on` | what `--patience` watches: `loss`, or `imputation`, the error on the entries `--hide` hid; see below | `loss` |
| `--hide` | hide this fraction of the training entries at random, score the model against the full counts, and report the error on the hidden entries at the end; the sample totals the model scales by are corrected for what the hiding took away (see below); needed for `--stop-on imputation` | nothing hidden |
| `--edge-balance` | how the `single` and `full` models score the pairs of samples that are not linked: `balanced` gives the linked pairs the same total weight as the absent ones, `none` scores every pair equally; see below | `balanced` |

**Running it**

| Option | What it does | Default |
|---|---|---|
| `--seed` | fixes the random starting point, so the same command gives the same model twice | unseeded |
| `--deterministic` | repeatable bit for bit on a GPU as well as on the CPU; needs `--seed` | off |
| `--threads` | compute threads | 4 |
| `--device` | `cpu`, `cuda`, or a particular card such as `cuda:0` | `cpu` |

**Starting from a model already fitted**

| Option | What it does | Default |
|---|---|---|
| `--from-model` | a saved model to start from instead of from scratch; it must have the same genes and the same shape, and the command says which differs if not | none |
| `--freeze` | hold part of the model still while the rest trains, for example `--freeze encoder`; may be given more than once; only with `--from-model` | nothing frozen |

Output on the vignette data, with `--patience 5`:

```
fitting a simple model: 200 genes, 36,090 parameters, 80 epochs
finished after 8 epoch(s)
  training loss   1456.05 -> 947.48
  validation loss 1119.80
  stopped early, best epoch was 3
wrote model.pt
```

Validation close to training means the model is not simply memorising. A
validation loss far above the training loss means it is, and the vignette's
toy of 120 samples gets there within a few epochs, which is why it runs with
`--patience`.

### Worth knowing

**Set `--seed` if you want the same model twice.** Without it the same command
gives a different model each time. On a GPU, add `--deterministic` as well.
Without it two runs of a graph model from the same seed differ at the fourth
decimal place, because one of its operations adds numbers in whichever order
the card finishes them; with it the two runs match bit for bit, measured on an
A100. The simple model matches on a GPU either way. Deterministic kernels can
be slower, which has not been measured, so it is off unless asked for.

**`--likelihood ZINB` is for counts that have more zeros than the negative
binomial can account for.** The negative binomial already produces plenty of
zeros on its own when a gene is lowly expressed, so a high proportion of them
is not by itself a reason to change. The zero-inflated form adds a separate
probability that a count is zero for a reason unrelated to expression level,
such as the transcript never being captured, and it costs one more parameter
per gene and sample along with the layer that produces it: on the small
example in the tests, 8,126 parameters against 7,514.

The PREFFECT paper found little evidence that fRNA-seq
needs it, which is why `NB` is the default. To ask the question of your own
counts rather than inheriting that answer, fit both distributions to them and
compare: [Choosing a distribution](choosing-a-distribution.md). The choice is
saved with the model, so `apply` rebuilds the distribution the model was
fitted with and does not need telling again.

**Set `--threads` to suit the size of your data.** PyTorch otherwise starts one
compute thread per core, which on a large shared machine can be over a
hundred, and coordinating them costs more than the arithmetic. The default of
4 suits a small example. Measured on 528 samples by 18,790 genes, in seconds
per epoch: 1 thread 3.9, 4 threads 2.1, 8 threads 1.3, **16 threads 1.2**, 32
threads 1.6. On small data one thread wins.

**`--validation` is optional, and leaving it out is honest.** A cohort too
small to split should not use a copy of the training data as its validation
set; without one, no validation loss is computed and none is reported.

**`--genes` chooses once, on the training data.** The validation set gets the
same genes, matched by name, never re-chosen. The saved model records the
names, and `apply` matches on them.

**`--patience` stops a run that has stopped improving, and it stops early.** It
watches the validation loss, or the training loss when there is no
`--validation`, and stops once that has not fallen by at least `--min-delta`
for `--patience` epochs in a row; the model written is the best epoch's, not
the last. Measured on a cohort of 220 samples at the default learning rate,
six seeds, a fifth of the samples held out: the validation loss
is lowest at about epoch 450 and within five per cent of that until 4,000, so
any patience stops at 240 to 830 epochs. That is the right place to stop if
the model is for samples it has not seen. It is ten times too soon if the
model is for the cohort it was fitted on: imputation of that cohort's hidden
entries is 1.5 times worse at epoch 300 than at 3,000 and is still falling at
16,000, and the loss that `--patience` watches cannot see it. Clustering
recovery of subtype from the latent is at its level by epoch 500 to 750
whichever you do, and past 7,000 the latent of unseen samples gets worse. **At
this learning rate, 3,000 to 5,000 epochs serves the loss and the latent within
a few per cent and in-cohort imputation within about a tenth, and 16,000 is
harmful for anything applied to new samples.** A rule that watches imputation
of the hidden entries instead, which lands near that range on its own, is
below; the curves are in the decision of 2026-10-28 on training length.

**`--stop-on imputation` stops where imputation of the fitted cohort settles.**
With `--hide 0.2` a fifth of the training entries are hidden from the model at
random, and after every epoch the model's error on those entries is measured,
the same number `exp/10` and `exp/11` score; `--patience` then counts epochs
without that error falling, and the model written is the best epoch's by that
measure. On the real cohort, replayed in `exp/11`, a patience of 500 to 2,000
epochs stops at 2,000 to 4,000 and keeps a model within four to six per cent of
the best imputation, where the loss rule keeps one from epoch 300 with 1.7
times the error. Two things to know. The objective scores the model
against the full counts, hidden entries included, so this measures how far the
fit of the cohort has gone, not how the model generalises to samples it has
not seen; for that, stop on the loss with `--validation`. And measuring it
costs one extra pass over the training data per epoch, about 70 per cent more
time an epoch on a GPU; the random stream is restored afterwards, so the run is
the same run as without the rule, stopped sooner or later. Run live on the real
cohort (`exp/12`), the rule stopped at 2,100 to 6,400 epochs on its own and
kept a model ten per cent better at imputation than a fixed 4,000-epoch run
in every seed, with the latent space level or better, at 1.7 times the time; the
decision of 2026-10-28 on stopping has the table. `--min-delta` applies to
whichever quantity is watched, in its own units. With the default of zero the
imputation rule waits until the error stops falling at all; on data where it
keeps creeping down (the synthetic data in `exp/12` fell two per cent over
4,000 epochs and the rule ran to its cap in four of six seeds) set
`--min-delta` to the smallest
improvement worth waiting for, about one per cent of the error: 0.005 on an
error near 0.5, 0.01 on one near 1.

**`--hide` corrects the sample totals for what it hid.** A sample's library
size is its total count, and the model multiplies everything it predicts for
that sample by it. Hiding a fifth of the entries takes about a fifth off that
total, while the counts the model is scored against keep the originals, so
without a correction every predicted count is short by the hidden fraction and
no amount of training recovers it: the composition sums to one across genes, so
the library alone sets the scale. The correction is each sample's gene count
divided by the number of its entries the model was shown, which reads how many
entries were hidden and never which values. It applies only to `--hide`, which
hides entries at random; nothing else about a run changes, and a run without
`--hide` is unaffected.

**`--edge-balance` is what lets the graph models learn their network on one
dataset, and has not been shown to on any other.** A sample-sample network
links each sample to a few others out of hundreds, so 92 to 99.8 per cent of
the pairs are not edges, and scoring every pair equally, which
`none` does, leaves the network half of the
model with nothing to learn: measured in `exp/17` on synthetic data with a
known network, it settles on predicting no edge anywhere in every seed, and
the loss cannot show it because that is exactly the prediction the loss
rewards. `balanced`, the default, gives the linked pairs the same total weight
as the absent ones; on that synthetic network the predicted network then ranks
the true edges as well as a predictor that knows the sample groups, and
clustering, imputation and the expression terms are unchanged. A third way,
scoring the linked pairs against
as many absent pairs drawn at random each step, did about as well more slowly
and lost; it lives in `exp/17`, not in the package.

**That recovery has not reproduced on either dataset run since.** On the DCIS
cohort at 773 genes, `exp/17`'s run of 2026-11-02 found every arm, `balanced`
included, ending with the decoder predicting exactly one half for every pair,
so the term made no difference there in either direction. On
`dataset_full_1`, `exp/18` found every decoder predicting no edge anywhere
under `balanced`, which is the outcome it exists to prevent. Nothing measured
is worse under `balanced`, which is why it is still the default, but do not
expect a usable network out of a graph model without checking the ranking with
`network_recovery` on your own data.

Two further things to know. Under
`balanced` a probability above one half is
not an edge call: equal weight moves the point where the model is undecided,
so most pairs come out above a half and the useful output is the ranking,
which is how the guide's `network_recovery` scores it. And `none` is there for
repeating a result produced before 2026-10-30, which every graph-model result
in this repository was. The decision of 2026-10-30 on weighting the network
term has the tables.

**`--batch-size` matters more for the graph models.** A graph model can only
attend to network edges between samples that are in the same batch, so a
batch much smaller than the cohort hides most of the network from it. For
`single` and `full`, consider a batch size equal to the number of samples.
How much this costs has not been measured.

**A part that does not exist cannot be frozen.** `--freeze encodr` is refused,
with the parts that do exist listed, rather than freezing nothing while
appearing to.

## `preffect apply`

Runs a saved model over counts and writes the cleaned result.

```bash
preffect apply --model model.pt --data my_study/new --out cleaned.h5ad
```

| Option | What it does | Default |
|---|---|---|
| `--model` | the file `train` wrote | required |
| `--data` | folder of counts to apply it to | required |
| `--out` | the `.h5ad` file to write; checked for writability before the model is loaded | required |
| `--tissues` | how many tissue files the data folder holds; only consulted if the model did not record its own count, and a model saved by this version always does | 1 |
| `--threads` | compute threads, as for `train` | 4 |
| `--device` | `cpu`, `cuda`, or a particular card such as `cuda:0`, as for `train` | `cpu` |

Output on the vignette's held-out split:

```
loaded a simple model written by PREFFECT 0.2.0.dev0, applying on cpu
matched 200 genes to the ones the model was fitted on
wrote cleaned.h5ad: 40 samples, 200 genes
```

The result carries the cleaned counts in `X`, the predicted proportions in
`layers['composition']`, and the latent summary of each sample in
`obsm['latent']`.

**Genes are matched by name**, so the columns of the file you apply to need
not be in the order you trained on. A gene the model was fitted on and the
file lacks stops the command and says which, rather than shifting everything
along by one.

`apply` will run over any counts, including the ones the model was trained on.
The only honest held-out result comes from data that was in neither `--train`
nor `--validation`.

## `preffect gui`

Opens [the graphical interface](gui.md) in your browser: check a folder of
counts, apply a saved model, and look at the figures. It answers on this
machine only, to `localhost`, and sends nothing anywhere; from a laptop,
forward the port over SSH.

```bash
preffect gui
```

| Option | What it does | Default |
|---|---|---|
| `--port` | the local port to serve on | 8501 |

It needs the `gui` extra, and says so if that is missing.

## `preffect example-data`

Writes a small simulated cohort, so the commands above can be tried before
any real data is to hand, and so the [worked example](vignette.md) runs from a
tool install, which has no copy of the source.

```bash
preffect example-data --out example_data
```

| Option | What it does | Default |
|---|---|---|
| `--out` | the folder to write under | `example_data` |
| `--seed` | seed for the simulation; the same seed writes the same files | 0 |
| `--unmatched` | the share of patients whose second tissue is left out of the full model's data, so there is something to pad; at least 0 and below 1 | 0 |

It writes three things, each as three splits with no sample in common:
`simple/` for the simple model; `single/`, the same counts plus a
sample-sample network with real edges in it; and `full/`, two tissues per
patient under different names with the correspondence table the full model
needs. 200 genes, three groups of samples that each raise a different block
of genes, negative binomial counts. It checks that no two splits share a
sample or are the same file before it reports success, and refuses to write
otherwise. At `--unmatched 0` it draws no random numbers for the
gaps, so the files are the same whether the option exists or not.

Three further things are true of what it writes, and each exists so an
example can show something a real cohort shows. **The genes are drawn once
per cohort**, so a cohort's three splits are the same genes and a model
fitted on one can be applied to another. **Batch B's own genes are raised** by a factor of 1.3,
a fifth of them picked at random and listed in the file under
`uns['batch_shifted_genes']`, which is a nuisance for the correction
variables to take out and is deliberately weaker than the groups.
**Every file carries the count each entry was drawn from**, in
`layers['true_expected_counts']`, so a filled-in entry can be scored against
the signal rather than against one noisy draw of it.

## When it refuses

The message names what was asked for and what to do instead. A few you may
meet:

| It says | It means |
|---|---|
| `no such folder` | the path given to `--train`, `--validation` or `--data` does not exist |
| `no files matching '*tau_<n>.h5ad'` | the folder exists but holds no count file named the way PREFFECT expects |
| `gene name(s) appear more than once` | gene names are the key everything matches on, and some repeat; it lists them, worst first |
| `the validation data is the training data` | printed by `check`, not a refusal: your two splits hold the same samples |
| `a network with NO edges between samples` | the file carries a sample-sample network with nothing in it; use `--kind simple` |
| `cannot write --out` | the output path was checked before any work and cannot be written |
| `--freeze only makes sense with --from-model` | there is nothing to hold still in a model that has not been fitted |
| `count(s) are not whole numbers, at ...` | from `prepare`: the counts table holds normalised or log-transformed values; it names the entries |
| `links every sample to every other` | `--neighbours` is one short of the sample count or more, which says nothing about which samples resemble which |
| `already in that folder` | `prepare` and `network` refuse to write over a file; give another `--out` or pass `--overwrite` |
| `already carries a sample-sample network` | `network` will not rebuild a real one unless `--replace` says so |

If a message tells you only that something is wrong and not what to do about
it, that is a defect in the message, and worth reporting.

## What the command line does not reach

Imputation scored against a genuine held-out set, clustering the latent
summary, and figures are done from Python. [Using it](guide.md) covers them.
