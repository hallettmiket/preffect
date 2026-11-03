# Choosing a distribution

This page is about deciding, before you train anything, whether your counts
need the zero-inflated distribution or whether the plain one is enough. It
covers the call that makes the decision, what the numbers mean, where it
refuses to answer.

## The question

PREFFECT can fit counts with either of two distributions. The **negative
binomial** says a gene has an average level and a spread around it wider than
pure counting noise. The **zero-inflated negative binomial** is the same thing
plus a separate probability that a count is zero for a reason unrelated to
expression level, such as the transcript never being captured.

Adding that probability always fits the data at least as well, because it can
be set to nothing and recover the plain distribution. So fitting better is not
a reason to use it. The question is whether it fits enough better to be worth
the extra parameter, and the Akaike information criterion is the usual way to
answer it. The criterion is the fitted log-likelihood penalised by the number
of parameters, and the smaller value wins.

The PREFFECT paper asked exactly this question of
fRNA-seq counts and found that the plain distribution is usually enough, which
is why [`Likelihood`](reference/config.md) defaults to it. The code here is the
estimator that produced that answer, ported so the evidence can be reproduced
and asked again of a new cohort.

## One gene

`compare_fits` takes one gene's counts across samples and returns both fits
with the verdict.

```python
import numpy as np

from preffect.zinb_fit import compare_fits

rng = np.random.default_rng(0)


def draw(mean, dispersion, zero_probability, n=400):
    counts = rng.negative_binomial(
        dispersion, dispersion / (dispersion + mean), size=n
    ).astype(float)
    counts[rng.random(n) < zero_probability] = 0.0
    return counts


plain = draw(mean=30, dispersion=3, zero_probability=0.0)
comparison = compare_fits(plain)
```

```text
>>> comparison.preferred
'NB'
>>> round(comparison.aic_difference, 2)
-1.33
>>> comparison.negative_binomial
NegativeBinomialFit(mean=30.97750020843003, dispersion=2.754773927681657,
                    log_likelihood=-1701.9800050052718, aic=3407.9600100105436,
                    dispersion_at_bound=False)
```

The same call on counts that really do have extra zeros:

```python
inflated = draw(mean=30, dispersion=3, zero_probability=0.4)
comparison = compare_fits(inflated)
```

```text
>>> comparison.preferred, round(comparison.aic_difference, 1)
('ZINB', 306.5)
>>> comparison.zero_inflated
ZeroInflatedFit(mean=30.300878641209515, dispersion=3.418954422277129,
                zero_probability=0.38475435143728937,
                log_likelihood=-1290.230705904805, aic=2586.46141180961,
                dispersion_at_bound=False, starts_converged=8)
>>> comparison.zero_inflated.observed_mean
18.642483731630936
```

## Reading the numbers

Four of the reported quantities need saying out loud, because three of them
are easy to read as something they are not.

- **`aic_difference` is the plain fit's criterion minus the zero-inflated
  one's**, so positive means the extra zeros earned their parameter. A
  difference under 2 means the two fit about equally well. Above 10 the
  evidence is usually called strong.
- **`mean` is the mean of the negative binomial component, not of the
  distribution**. Above, the component's mean is 30.3 while the counts
  themselves average 18.6, because two counts in five were forced to zero.
  `observed_mean` is the second number.
- **`dispersion` is PREFFECT's `theta`, the inverse dispersion**. Large means
  close to Poisson, small means heavily overdispersed, and the variance is
  `mean + mean**2 / dispersion`. A textbook that calls the dispersion `alpha`
  means `1 / theta`.
- **`dispersion_at_bound` means the counts are at least as tight as a
  Poisson**, so the dispersion is not identified and what is reported is the
  largest value the fit allows. This is common on genes with almost no counts
  and on anything close to constant.

## A whole matrix

`compare_fits_over_columns` runs the same comparison down every column of a
counts matrix, which is one gene each in a samples-by-genes matrix, and
returns a table with one row per gene.

```python
from preffect.zinb_fit import compare_fits_over_columns

matrix = np.column_stack(
    [plain, inflated, draw(5, 1, 0.0), draw(80, 6, 0.6)]
)
table = compare_fits_over_columns(
    matrix, names=["GENE_A", "GENE_B", "GENE_C", "GENE_D"]
)
```

```text
>>> table[["name", "zero_fraction", "preferred",
...        "aic_difference", "zinb_zero_probability"]].round(3)
     name  zero_fraction preferred  aic_difference  zinb_zero_probability
0  GENE_A          0.002        NB          -1.332                  0.002
1  GENE_B          0.385      ZINB         306.510                  0.385
2  GENE_C          0.188        NB          -1.364                  0.037
3  GENE_D          0.565      ZINB         429.242                  0.565
```

`GENE_C` is the row worth looking at. Nearly a fifth of its counts are zero
and the plain distribution still wins, because a gene with a mean of 5 and a
wide spread produces that many zeros on its own. A high proportion of zeros is
not evidence of zero inflation by itself, which is the whole reason for fitting
rather than counting.

A `DataFrame` brings its own column names, so `names` can be left out when you
pass one.

## Where it refuses to answer

Three situations raise rather than returning a number, because a number would
be a guess.

| What happened | What is raised | Why |
|---|---|---|
| The values are not whole, non-negative, finite numbers | `DataError` | A count distribution has no probability to give them. Round the counts yourself if that is what you mean, as the paper's notebooks did. |
| Every count is zero | `FitError` | Any mean fits equally well as long as the extra-zero probability is one, so nothing is identified. |
| No search converged | `FitError` | There is no fit to report. |

In a sweep over thousands of genes, stopping at the first gene that cannot be
fitted is the default, because a sweep that quietly drops genes turns a claim
about a cohort into a claim about whichever subset happened to fit. Pass
`on_failure="record"` to carry on instead; the reason goes in the row's
`problem` column and its numbers are left missing.

## How long it takes

Fitting both distributions to one gene across 220 samples takes about 0.18
seconds on one thread. So 2,000 genes take six minutes and a 19,000-gene
cohort takes about an hour. Nothing here is on a training path, so it has not
been made faster than that.

If you are sweeping a whole cohort, the reason to pass `on_failure="record"`
is that an hour's work should not be thrown away by one gene. Read the
`problem` column afterwards and decide what to do about the rows that have
one.

## Reproducing the paper's comparison

The reference values are in `tests/data/version_0_1_zinb_fits.json`: nine
count vectors with the fits the paper's notebooks compared against.
`tests/test_zinb_fit.py` checks the estimator against them.
