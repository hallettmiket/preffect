# Changes

## 0.2.0

Version 0.2 is a rewrite of version 0.1, not a patch. Three things follow for
anyone moving from 0.1:

- **The Python interface is new.** `configs`, `factory`, `Inference` and
  `Cluster` are gone. Settings are typed objects (`PreffectConfig`), and the
  command line (`preffect check`, `train`, `apply`, `gui`) covers the common
  work. The documentation's getting-started page shows both.
- **Models saved by 0.1 cannot be loaded.** PyTorch 2.6 and later refuse the
  pickled sessions 0.1 wrote, and 0.2 saves a versioned checkpoint of tensors
  and settings instead. Refit.
- **A seeded 0.2 run will not match a seeded 0.1 run bit for bit**, even where
  nothing in the mathematics changed, because 0.1 initialised layers it never
  used, and that consumed random draws before the layers that are used were
  initialised. Compare the two versions across several seeds, not one.

**These fixes may change your results from version 0.1.** Every item in the
first group below alters the trained model or what it returns; the second
group alters reported numbers only. Read the first group to see which fix
applies to your run: for example, if your sample count was not a multiple of
the batch size, version 0.1 never trained on the last partial batch.

### Fixes that change the trained model or its output

- The full model could not be constructed in the public 0.1 release: its graph decoder was built without a required argument, so every attempt raised a `TypeError`.
- Requesting the `single` model silently built the `simple` one; the sample-sample network was loaded and then ignored.
- The library-size term was computed, logged and left out of the objective for the `single` and `full` models.
- Per-tissue loss weights were paired with the losses by `zip`, so a short weight list dropped a tissue from the objective without a word.
- The loss divisor counted terms the chosen model did not have, so reported losses from different models were scaled differently and could not be compared.
- The last partial batch of every epoch was discarded, so samples beyond the last whole batch never entered training.
- Genes were ranked and cut separately in each split, so a column meant a different gene in training and validation. Genes are now chosen once, from the training data, and matched by name everywhere.
- Inference ran through the sampling path, so the same model over the same data returned different cleaned counts each time. Inference now uses the latent mean.
- Settings lived in one module-level dictionary that several parts of the code modified, so a second model built in the same session inherited the first one's settings.
- Asking for more samples than the cohort holds silently switched on sampling with replacement.
- Five shipped defaults disagreed with the paper's own parameter table; library-size inference was on where the table says off.
- Two places ignored a request for the CPU; `--device` is now honoured by `train` and `apply`.
- The adjacency decoder's width followed the batch size, so the architecture changed with a training setting and could not take a short final batch. It is now an inner-product graph decoder that takes any batch size.
- The expression encoder was fed raw counts, which reach millions in a real cohort, so one ordinary optimiser step could move its output a thousandfold: the single-epoch loss spikes. It is now fed log(1 + counts); the targets, decoder and likelihood still see raw counts.
- The network term was a plain mean over every pair of samples, of which 98 to 99.8 per cent are not linked, so the network half of the model learned nothing. Linked and absent pairs now carry equal total weight; `--edge-balance none` reproduces the old term.
- Under a mask, the library size was the row sum of the counts with entries hidden, so every prediction for that sample was short by the hidden fraction and nothing in the model could recover it. The total is now corrected for the number of entries hidden. A run without hidden entries is unchanged.
- Transfer learning froze nothing: both branches of its condition set every parameter trainable. Freezing now works, and asking to freeze a part that does not exist is refused.
- Masked training trained the model toward the masked matrix, teaching it to predict zero at every hidden entry. It is now scored against the original counts.
- A leaky rectifier on the latent mean and log-variance, and a second one after the decoder, were removed. Measured across seeds, neither changes what the model reconstructs.

### Fixes to reported numbers

- The validation loss was divided by the number of tissues rather than the number of batches: a sum reported as a mean, not comparable across batch sizes.
- A validation set was mandatory, so cohorts too small to split used a copy of the training data and reported a training loss as validation. It is now optional, and a validation set that repeats the training samples is reported.
- The relative imputation error divided by zero at zero-truth entries, and its count of excluded entries was taken over the whole matrix rather than the scored entries.
- Three defects in the figure and clustering code, and a PAM50 heatmap whose gene list had two namings with the switch between them in the wrong place.
- The standalone zero-inflation estimator worked out the dispersion by a division nothing kept positive and returned nothing where it failed; every search started from the sample mean and variance whatever zero inflation it was exploring; and the optimiser's success flag was read as convergence. It now varies the dispersion directly, starts each search consistently, and raises rather than returning nothing.

### Mistakes made impossible

- The check on input data could not fail. It can now, and `preffect check` runs it without fitting anything.
- Two samples with the same name are refused rather than merged.
- A sample-sample network with no edges is refused. Every dataset shipped with 0.1 carried an identity matrix there, so a graph model on them cost ten times the parameters for nothing.
- Gene selection matches by name only; a panel that matches nothing raises.
- A model kind that cannot be honoured raises and names the setting, instead of being rewritten to another kind.
- Imputation error needs real truth; 0.1 invented one from the per-gene mean when none was present.
- A dead configuration key and a dead method that would have crashed if anything had called it are gone.
- Settings are frozen once built; to vary one, `config.replace(...)` returns a new, re-checked set.

### Usability and speed

- The command-line entry point in 0.1 ran nothing. The command line now has `check`, `train`, `apply` and `gui`, and `--likelihood {NB,ZINB}` chooses the distribution, which nothing in 0.1 could do from outside its settings file.
- Nothing in 0.1 helped you build its input file; the layout was described and left to be assembled by hand. `preffect prepare` writes it from a counts table and a sample table, with the correspondence table the full model needs, and `preffect network` adds a sample-sample network to files that have none, leaving the files it read untouched. Both refuse rather than repair, naming the entries at fault. `preffect.prepare` holds the same functions for use from Python, and its network builder reproduces the one the project measured edge for edge.
- Saved sessions were unreadable on PyTorch 2.6 and later. Checkpoints are now versioned files of tensors and settings.
- A saved model carries the record of the run that made it: the loss at every epoch for each split, the epoch it stopped at and why, and how long the fitting took. Nothing in 0.1 kept any of it, so how a fit went lived only in what the training run printed to the terminal. It travels in the checkpoint's free-form `extra`, so the format version does not move and a file written before it was kept still loads.
- The graphical interface reads as a report rather than as one long page: a page describing the run, with its loss curve and the settings that change what the numbers mean, and then one page per figure. Each figure page opens by saying what is on the axes, what a good picture looks like and what to do about a bad one.
- The interface also looks up one gene or one sample, ranking the cohort worst fit first by the mean log probability per gene of each sample's counts; hides a fraction of the loaded counts and scores the model on filling them back in, the way `preffect train --hide` reports at the end of a run; and puts two saved models side by side on the same counts, figure by figure on one set of axes, with a table that names which is better on what and refuses to name one where the difference is below a stated threshold.
- The interface writes the input file as well as reading it: upload a counts table and a sample table, or name their paths, say which way round the counts are written, optionally build the sample-sample network with the summary redrawn as the neighbour count moves and optionally the table saying which sample is which patient, and it writes one file per tissue without overwriting anything and finishes on what `preffect check` says about them.
- A counts or sample table that could not be opened reached the caller as a pandas traceback rather than a sentence, on the command line and in the interface. Both readers now refuse by name, for a path that is not there and for a file that is not a table.
- No run was reproducible. `--seed` makes a run repeatable on the CPU, and `--deterministic` makes it repeatable bit for bit on a GPU as well.
- PyTorch used every core and was several times slower for it; the default is now four threads. Setting a thread count after the model was built changed the result silently; it is now set first.
- About 95 per cent of 0.1's epoch time went on one log diagnostic and a plot. On the same CPU, 0.2 is five times faster per epoch than 0.1 with those stubbed out, and sixty times faster than 0.1 as shipped.
- The simple model allocated a sixteen-million-parameter graph attention layer it never used.
- A patient missing one tissue is padded, as 0.1 padded them, with the padding recorded in the data rather than recovered from a name, refused as a masking target, dropped from the output and counted by `preffect check`.
- `preffect example-data` writes the simulated example cohort, so an install from GitHub, which has no copy of the source, can run the worked example; a notebook in `vignettes/` walks the same steps and is run by the test suite.
- The example cohort's genes are drawn once per cohort and shared by its three splits, so a model fitted on the training split can track the test split; batch B's own genes carry a real 1.3-fold lift for the correction variables to take out; and every file carries the count each entry was drawn from, in `layers['true_expected_counts']`, so filled-in entries can be scored against the signal rather than against one noisy draw of it. Nothing about a real cohort changes.
- The package installs with `pip` or `uv`; settings are typed and validated; the API reference is generated from the docstrings; and the suite of more than a thousand tests, most guarding against one of the defects above, runs in continuous integration.

### Not ported

- Differential expression, which 0.1 itself marked "in development".
