# The graphical interface

A page in your browser for the part of the work you want to look at: turning
your two tables into the file PREFFECT reads, whether that folder of counts
can be fitted and what it holds, how the fit that produced a saved model went,
and the figures that say whether to believe it. It reads like a report. The
pages sit under six headings in the sidebar, in the order the work is done,
and you move through them a page at a time. Past the figures it will look up a
single gene or a single sample, ask whether the model can fill in a count it
was not shown, and put two saved models side by side on the same counts.

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

## How it is laid out

The sidebar lists the pages under six headings, top to bottom in the order the
work is done. Each title is the question the page answers, and clicking it
opens that page and nothing else. A page is listed only once what it needs is
open, so before a run is open you see four pages, and the rest appear when a
model has been applied. Under the list, three lines say what is open, on every
page: the folder of counts, the model, and the second model if there is one.
Every page opens by saying what it is for, before any number or picture. Each
page has an address of its own, so a figure page can be bookmarked and
returned to.

| Heading | Pages | Needs |
|---|---|---|
| Start | Start here | nothing |
| Your data | Prepare the input file; Can this be fitted?; What does the data look like? | a folder of counts, for the last two |
| The fit | How did the fit go? | a run |
| Does the fit hold? | one page per figure, six in all | a run |
| Ask the model | What did it make of one gene or sample?; Can it fill in hidden counts? | a run |
| Compare | How do two models compare? | a second model, opened on the same counts |

## Start here

The landing page. It says in one sentence what the interface is for and that
fitting itself happens on the command line, then says whether a run is open,
and then asks for what it needs: the folder of counts and how many tissue
files are in it, the model `preffect train` wrote and the device to run it
on, and, optionally, a second model to compare. A folded section at the bottom
walks the four steps for a first visit, and stays folded after that.

**Nothing to open yet?** Press **Try it on the example data** and the
interface writes the simulated example cohort under the system's temporary
folder, fits the worked example's two models on it, a plain one and a
zero-inflated one, with the worked example's own settings, fills in the paths
it made and opens both. It takes about ten seconds. It is the one fit the
interface makes: the cohort is small and simulated, and the fit goes through
the same `preffect train` the worked example runs. Every page then has
something behind it, and the comparison page has two different models to put
side by side. Nothing is filled in until you press it, because a path that
only exists on one machine is not a default this package will ship.

**A run is a saved model and the counts you want to apply it to.** Give both
and press **Open the run**. Applying the model takes a moment and then happens
once: moving between pages after that does not refit or reapply anything, and
the line at the top of the page says what is open. The three pages under
*Your data* need only the folder, so you can prepare, check and look at a
cohort before you have fitted anything.

Name a second model and press **Open the comparison** and it is applied to the
same counts, which adds the *Compare* heading. Opening a different first run
drops the second, because two models over two different sets of counts is not
a comparison and nothing on the page would say so.

## Your data

### Prepare the input file

Writes the file everything else here reads, from the two tables you are
likely to have: a counts table with a count for every sample and gene, and a
sample table saying what is known about each sample. It is
[`preffect prepare`](command-line.md) with the steps laid out, and it is
where to start if PREFFECT has never seen your cohort.

Upload the two tables, or type their paths, which is usually right when
PREFFECT runs on a server you reached over an SSH tunnel and the files are
already there. Say which way round the counts are written: one row per
sample is assumed, and one row per gene, which is how many pipelines write
one, is flipped as it is read. The page then says how many samples and genes
it found and what columns the sample table has, so a table read the wrong way
round is obvious before anything is built.

Two things are optional:

- **A sample-sample network**, which only the single and full models use. A
  slider sets how many samples each one links to, and the summary underneath
  redraws as it moves: edges, links per sample, and samples left with none.
  Name a column of your sample table and it also gives the share of links
  joining two samples with that same label against the share that would join
  by chance. Two shares close together mean the network carries no biology.
- **The table saying which sample is which patient**, which the full model
  needs and which appears once you ask for two tissues or more. Choose the
  column naming the patient and the page says how many patients have a sample
  in each tissue and how many have none.

Name a folder, press the button, and one file per tissue is written. **Nothing
already in the folder is written over**: a name that is taken is refused, with
the path it would have replaced. Every count is checked before anything is
written and a problem names the entries at fault. The page finishes by running
the same check the next page runs, so you end on the sentence that says
whether what you made can be fitted.

### Can this be fitted?

What `preffect check` reports, and it needs no model. It says how many samples
and genes each tissue file holds, whether there is a sample-sample network and
how many links it has, and, if something would stop a run, the sentence the
run would have refused with. It is a pass or a fail. Run it on a new folder
before you spend hours fitting anything.

### What does the data look like?

What the folder holds, read without a model, for the decisions that come
before a fit. For each tissue file:

- how many samples and genes there are, and **how much of the matrix is
  zero**, which is what decides whether to ask `preffect train` for the
  zero-inflated distribution, `--likelihood ZINB`: a matrix that is largely
  zeros is where it earns its extra parameter, and one that is not usually
  does not need it;
- **the sample table's columns**, with how many distinct values each takes,
  how many are missing and a few examples, which is where a batch to correct
  for or a label to colour the latent summary by comes from;
- whether the file carries a **sample-sample network**, and with how many
  links, which only the graph models use;
- **how deeply each sample was sequenced**, as a histogram of each sample's
  total count. A bar far to the right of the rest is a sample sequenced far
  more deeply than the others and one far to the left is a failed library;
  either pulls a fit towards itself, and this is the moment to drop it, before
  any model has been pulled by it.

The page before this one says whether the folder can be fitted at all; this
one says what you would be fitting.

## The fit

### How did the fit go?

Describes the fit itself, and it is the page to read first once a run is open.
It is there so that nobody reads a figure without first knowing whether the
model behind it had finished moving. It shows, in this order:

- what was fitted: the kind of model, the count distribution it used, how many
  samples and genes were matched, and how many tissue files it holds
- the settings that change what the numbers mean: whether the library size was
  read from the counts or inferred by the model, the batch size and the
  learning rate
- how the fit went: how many epochs it ran out of how many were asked for,
  whether it stopped early, which epoch's weights were saved, how long the
  fitting took, and one sentence saying how the run ended
- **the loss curve**: the mean loss per batch at every epoch, with the
  validation curve on the same axes where there was a validation split
- a button to download the cleaned counts as an `.h5ad` file, the same file
  `preffect apply` writes

**A model saved by an older version of PREFFECT says so.** The loss curve, the
epoch count and the timing come from a record `preffect train` now writes into
the model file. A file written before that was kept is still a perfectly good
model: the page says it was saved before its history was kept, and everything
that does not need the history is shown as usual. Refit if you want the curve.

## Does the fit hold?

Each figure gets a page of its own, titled with the question it answers, and
each page opens by saying what the picture shows, what a good one looks like,
what a bad one looks like and what to do about it. They are written for
someone who has not read the code, and they come in order from the simplest
check of the fit to the latent summary:

| Page | What is drawn |
|---|---|
| Does it track sequencing depth? | each sample's observed total count beside the total the model inferred |
| Does it reproduce the counts? | for the most variable genes, the count the model fitted against the count observed, sample by sample |
| Did it remove the batch effect? | each gene's average in one batch against its average in another, in the data and in the model's view of it |
| Is the noise per gene believable? | gene averages, fitted dispersion, library sizes, and the dispersion against the average |
| Does it get the zeros right? | how often each gene is zero against the zero-inflated model's probability of a zero |
| Do similar samples group together? | the model's summary of each sample squeezed to two dimensions, coloured by a column of your sample table |

Two of these need something the model may not have. **Did it remove the batch
effect?** needs a column of your sample table holding one label per sample,
such as the run or the site it came from, and you choose which column on the
page. **Does it get the zeros right?** needs the zero-inflated distribution,
which is `preffect train --likelihood ZINB`. Where what is needed is missing,
the page still opens with its description and says in a sentence what is
missing and what to refit with, rather than drawing something empty.

Every figure here is one of [`preffect.plots`](reference/analysis.md)' figures,
drawn by the same call a script would make, so anything seen here can be
reproduced and put in a paper. The loss curve on the fit page is the one
exception, and it is drawn from what the training run recorded.

## Ask the model

### What did it make of one gene or sample?

Answers a question about one gene or one sample, where every page before it
describes the whole cohort at once. Pick a gene and it shows the count the
model fitted in each sample against the count observed, how closely the two
move together across the cohort, how variable the model thinks the gene is,
and how often it is zero. Pick a sample and it gives one number saying how
well the model accounts for that sample, and shows where that number sits
among the rest.

**The measure for a sample is the mean log probability per gene** of that
sample's observed counts under the distribution the model fitted for it.
Larger is better and every value is negative, because these are probabilities
of a particular count. It is taken per gene rather than summed over genes so
that it does not grow with the size of the panel.

The part worth your time is the list at the bottom, which is the cohort
ordered worst fit first. A sample at the top of it is usually a failed
library, a mislabelled sample, or a sample of a different tissue from the
rest, and it is worth looking at before it goes into anything downstream.

### Can it fill in hidden counts?

Asks whether the model can fill in a count it was not shown. It hides a
fraction of the entries of the counts you loaded, chosen by a slider and a
seed, runs the model on what is left, and compares what it predicted at the
hidden entries against what was really there. This is what
[`preffect train --hide`](command-line.md) reports at the end of a run, asked
of a model that is already fitted.

The number is the average relative distance from the observed count, so 0.5
means the typical filled-in entry is out by half of the count that was there,
and smaller is better. It comes with how many entries it was scored over and
how many were left out, because a relative error cannot be computed where the
true value is zero.

**Read it as a statement about this cohort, not about new samples.** The model
has already seen every sample here. Filling a gap for a patient it has never
met is a harder question, and it needs a held-out folder.

Where the file carries `layers['true_expected_counts']`, which only simulated
data can have, the page also scores against that, which is the better
comparison: an observed count is one draw from a distribution, so part of the
distance from it is noise nothing could have predicted. On the [worked
example](vignette.md) the same model is out by about 65 per cent of the
observed count and about 12 per cent of the count it was drawn from. Ordinary
counts carry no such layer, and PREFFECT will not invent one.

## Compare

### How do two models compare?

Appears once a second model has been opened on the same counts, so that the
difference you see is the model and not the data. It has three parts: what
each run recorded about itself, side by side; any figure page drawn under both
models on one set of axes; and a table of three numbers you might decide on,
worked out when you ask because clustering costs a minute on a real cohort.
Those three are the median correlation between fitted and observed across
genes, the latent clustering scored against a column of your sample table,
and the imputation error, which reuses the fraction and seed the page above it
used.

The last column of the table says which run is better on each measure, and
reads **neither** where the two differ by less than a threshold set for that
table: 0.01 for a correlation, 0.05 for the adjusted Rand index, and five per
cent for the imputation error.

**Those thresholds stop a tiny difference being read as a result. They are not
a measurement of the noise.** This page compares two particular fits, not two
methods: fitting either model again from a different seed gives a different
answer. Treat a small difference here as nothing at all.

## What it does not do

**It does not fit a model**, apart from the example on the landing page,
which is seconds on simulated data and goes through `preffect train` itself.
Training a real cohort is hours, and belongs on the command line where it can
be left running and logged. The interface reads
what `preffect train` wrote. That is why the count distribution is shown here
and chosen there, with
[`preffect train --likelihood`](command-line.md): a zero-inflated model and a
plain one of the same kind look alike on these pages and predict different
counts, so the page owes you the answer even though it cannot change it. The
same holds for everything else on the fit page: it reports the settings a run
used and changes none of them.

**It does not score anything on samples it has not seen.** The imputation page
hides entries of the counts you loaded and scores the model on putting them
back, which is a statement about this cohort and not about new patients. The
comparison page scores the latent clustering against one column you choose.
Anything past those two, a held-out folder or a decision about what to compare
against, is [Using it](guide.md) from Python.

## If something is wrong

A folder that cannot be read, counts that would stop a run, or a device the
machine lacks shows as a sentence on the page, the same sentence the command
line prints. A figure that cannot be drawn from this particular model says so
on its own page, in the sentence `preffect.plots` refuses with. If the latent
figure says it needs the `viz` extra, the `gui` extra includes it, so that is
an environment assembled by hand; install it.
