# Reading and preparing data

Building the input files, loading counts, checking they can be used, choosing
genes and samples, hiding entries for an imputation experiment, and reading
the sample-sample network.

The checks are worth running before anything else. `preffect check` is built
from them, and on real data they catch things worth knowing: repeated gene
names, a validation set that is a copy of the training set, and a network with
no edges in it.

## Building the input files

What `preffect prepare` and `preffect network` are made of: a sample-sample
network built from counts, a correspondence table built from a patient
column, and the writer that puts them in a file the rest of the package
reads. Every function takes plain arrays and data frames, so the browser
interface can call them with what somebody uploaded.

::: preffect.prepare

## Loading

::: preffect.data.loading

## Checks

::: preffect.data.checks

## Choosing genes and samples

::: preffect.data.selection

## Named gene sets

::: preffect.gene_sets

## Hiding entries

::: preffect.data.masking

## Batches

::: preffect.data.batching

## Tensors

::: preffect.data.tensors

## The sample-sample network

::: preffect.data.graph

## Several tissues

::: preffect.data.tissues

## Example data

::: preffect.example_data
