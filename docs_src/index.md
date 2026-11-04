# PREFFECT 0.2

## Description

PREFFECT (PaRaffin Embedded Formalin-FixEd Cleaning Tool) is a probabilistic
framework for the analysis of RNA sequencing data from formalin-fixed paraffin
embedded (FFPE) samples. PREFFECT uses generative models to fit distributions
to observed expression measurements while adjusting for technical and
biological variables. PREFFECT offers a series of models that exploit multiple
expression profiles generated from matched tissues for a single sample,
sample-sample adjacency networks and clinicopathological/technical sample
variables.

PREFFECT is designed to accomplish the following tasks:

- Accurately denoise RNAseq data from FFPE materials (fRNAseq)
- Impute missing values from the count matrix
- Improve performance of standard analyses

## Where to go

- **[Getting started](getting-started.md)** installs it and fits a model.
- **[The command line](command-line.md)** is every command and every option,
  with what each is for and when to change it.
- **[The graphical interface](gui.md)** prepares and looks at a folder of
  counts, applies a saved model and shows the figures a page at a time, in a
  browser, on your own machine.
- **[Using it](guide.md)** covers the real workflow: checking data, choosing
  genes, imputing, clustering, drawing.
- **[Reference](reference/config.md)** is the API, taken from the docstrings.
- **[What changed from 0.1](changes.md)**

## A note on this version

Version 0.2 is a rewrite, not a patch.
[`CHANGELOG.md`](https://github.com/hallettmiket/preffect/blob/main/CHANGELOG.md)
lists every fix and change relative to version 0.1, and opens with how to tell
which of them explains a difference in your results.

## Contributors

- **Eliseos J. Mucaki**, Department of Biochemistry, Western University, London, Canada
- **Wenhan Zhang**, Department of Biochemistry, Western University, London, Canada
- **Aryamaan Saha**, Department of Computer Science, Columbia University, New York, USA
- **Vanessa Dumeaux**, Department of Anatomy and Cell Biology and Department of Oncology, Western University, London, Canada
- **Michael T. Hallett**, Department of Biochemistry and Department of Oncology, Western University, London, Canada. Contact: <michael.hallett@uwo.ca>

The method is described in Mucaki *et al.*, *Journal of Translational Medicine*
23:1023 (2025), <https://doi.org/10.1186/s12967-025-07031-y>.

<p align="center">
  <img src="assets/logos/western_logo.png" alt="Western University" height="60"/>&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/nserc_2024.png" alt="NSERC" height="60"/>&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/cihr_color_logo_2024.png" alt="CIHR" height="60"/>
</p>
