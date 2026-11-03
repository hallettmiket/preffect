# What changed from version 0.1

This page is the summary. The point-form list of every fix, opening with how
to tell which one explains a difference between your version 0.1 and version
0.2 results, is
[`CHANGELOG.md`](https://github.com/hallettmiket/preffect/blob/main/CHANGELOG.md)
in the repository. The experiment folders named `exp/NN` and the decision
records mentioned below belong to the development record, which is not part
of this release.

## The ones that move a number

**The network term weights the linked pairs against the absent ones.** Version
0.1 scored the predicted sample-sample network against the observed one with a
plain mean over every pair, and carried a weight for present edges and one for
absent edges that nothing set. On a network one per cent dense that is 99 parts
"no edge here" to one part "edge here", and measured on synthetic data with a
known network, the network half of the model learned nothing under it: the
decoder settled on predicting no edge anywhere and the network latent collapsed
onto its prior, in every seed. The linked pairs now carry the same total weight
as the absent ones, which on that synthetic network recovers its structure as
well as a predictor that knows the sample groups, in six seeds of six, with
clustering, imputation and the expression terms unchanged. On the two datasets
run since, the DCIS cohort in `exp/17` and `dataset_full_1` in `exp/18`, the
balanced term did not recover a network: on the first every arm's decoder ended
at exactly one half for every pair, and on the second every decoder predicted
no edge anywhere. Nothing measured is worse under it, which is why it stays the
default. `--edge-balance none` gives the
old term, for repeating a result produced under it. Decision of 2026-10-30.

**The objective is one list of named terms.** Version 0.1 kept two hand-written
lists, a numerator and a divisor, and they disagreed: a term was computed,
weighted, logged, and then left out of the sum while its weight stayed in the
divisor. The total and the divisor are now derived from the same list, so they
cannot disagree.

**Per-tissue weights are checked, not truncated.** Version 0.1 paired weights
against losses with `zip`, which stops at the shorter, so a two-tissue run with
a one-entry weight list dropped the second tissue out of the objective. A list
of the wrong length now raises and names every offending weight.

**The reported validation loss is a mean.** Version 0.1 divided by the number of
tissues rather than the number of batches, so with six validation batches the
figure was six times the mean and not comparable across batch sizes.

**The last partial batch of each epoch is kept.** Version 0.1 discarded it.

**Inference uses the latent mean.** Version 0.1 ran inference through the
training path, which samples, so the same model over the same data gave
different cleaned counts every time.

**Two rectifiers were removed**, one on the latent parameters and one after the
decoder. Neither changes what the model reconstructs. These were decided on
synthetic data at 150 genes and the decision records say so.

**The encoder is fed log(1 + counts).** Version 0.1 fed it raw counts, which
reach millions in a real cohort, so an ordinary training step could move the
encoder's output a thousandfold at once: the loss spikes of issue #22. On the
log scale the spikes are gone in every seed measured, imputation error more
than halves and clustering recovery more than doubles. What the model
predicts, and the scale it is scored on, are unchanged. Measured on the simple
model and on the one-tissue graph model, with the same result on both, and
measured again on 2026-11-02 under the corrected library size, which left the
result standing.

**The library size is corrected for entries a mask hid.** A sample's library
size is its total count, and the model multiplies everything it predicts for
that sample by it. Both versions read that total off the counts the model was
shown, which under a mask is short by the hidden fraction while the counts the
model is scored against are not, so every expected count came out short by the
same fraction and nothing in the model could recover it. The total is now
estimated back from how many of a sample's entries were hidden, never from
their values. Measured on synthetic counts with no zeros, a fifth hidden: the
expected counts went from 0.801 of the true grand total to 1.001. A run
without hidden entries is bit for bit what it was, so nothing `apply` produces
for a real cohort moves. What it does to a reported imputation error depends
on how well the composition was fitted and is not one factor. The end-to-end comparison
of the two versions was measured again on 2026-11-02 with both of them
corrected, version 0.2 inside its model and version 0.1 as its stored
predictions are scored, and it left every verdict standing. Finding 56.

**The full model trains.** Until 2026-11-02 it could be built, run forward and
scored, but `train` stopped at its first batch, because the loop asked every
batch a question only a single-tissue one answers. All three models now train
from Python and from the command line, and the suite trains each of them.
Finding 55.

**A patient missing one tissue is padded, and the padding is accounted for.**
Version 0.1 kept such a patient and filled the tissue they lacked with the
rounded per-gene mean of the real samples, a row it called a ghost. Version
0.2 refused them outright until 2026-11-02 and now does what version 0.1 did,
rule for rule, with three things version 0.1 left implicit made explicit: the
data carries which rows are padding rather than recovering it from a name
beginning `tau_`, a mask that would hide an entry in a padded row is refused
rather than silently skipped, and `infer` and `apply` drop padded rows from
their output and say how many. `preffect check` reports how many patients lack
each tissue before anything is fitted. A cohort with nothing to pad trains and
scores exactly as it did. The decision of 2026-11-02 on unmatched samples has
the measurement against the alternative, which is dropping those patients.

## The ones that stop you doing something

These make a class of mistake impossible rather than changing a result.

- A model kind that cannot be honoured **raises**. Version 0.1 silently turned
  a request for the graph model into the plain one, loaded the network, and
  ignored it.
- `imputation_error` **requires truth**. Version 0.1 invented it as the
  per-gene mean when it was absent.
- Gene selection **matches by name**, never by position, and a panel that
  matches nothing raises.
- A sample-sample network **with no edges raises**. Every dataset version 0.1
  shipped carries an identity matrix there (findings 35 and 54), so a graph
  model on them was ten times the parameters for nothing.
- The settings are **frozen**. Version 0.1 had four parts of the code modifying
  a shared dictionary while it ran.

## What is ported from the paper's own tools

**The standalone estimator that decides which distribution a cohort needs.**
Version 0.1 shipped it as `wrappers/_ZINBEstimator.py` and the paper used it
to establish that the plain negative binomial is enough for fRNA-seq, which is
why the plain one is the default here. It is now
[`preffect.zinb_fit`](choosing-a-distribution.md), which fits both
distributions to one gene's counts and reports which the Akaike information
criterion prefers, and which runs over the columns of a matrix in one call.

The likelihood is unchanged and the two versions agree on every set of counts
version 0.1 could fit. Three things about the search changed, each because
version 0.1 lost fits it should have kept: it varies the dispersion rather
than dividing for it afterwards, it starts each search consistently with the
amount of zero inflation that search is exploring, and it raises rather than
returning nothing. Findings 57, 58 and 60 and the decision of 2026-11-02 on this
estimator have the measurements.

**Choosing the zero-inflated distribution from the command line.** The
distribution, the model paths that use it and the loss were all ported and
tested, and then there was no way to ask for it: `preffect train` had no
likelihood option until 2026-11-02. Version 0.1 documented
`model_likelihood: NB | ZINB` in its settings page, and nothing recorded a
decision to drop it, so it was an omission rather than a choice. It is now
`preffect train --likelihood {NB,ZINB}`, the choice is saved with the model so
`apply` rebuilds the right distribution, and the graphical interface shows
which distribution a loaded model was fitted with. Finding 61.

## What is not ported

A differential expression function that version 0.1 itself marks "in
development". Transfer learning is ported (`--from-model` and `--freeze` on
the command line, `preffect.pretraining` in Python), and it now freezes what
it is asked to freeze: version 0.1's version set every part trainable
whichever branch ran.
