# The three models

`simple` uses counts alone. `single` adds one sample-sample network. `full`
takes several matched tissues.

Asking for one and getting another is not possible: a request that cannot be
honoured raises rather than quietly becoming a different model.

## Layers

::: preffect.layers

## The simple model

::: preffect.model

## Graph layers

::: preffect.graph_layers

## The single-network model

::: preffect.graph_model

## The full model

::: preffect.full_model

## Distributions

::: preffect.distributions

## Choosing between them

Fitting both distributions to one vector of counts and comparing them, which
is how the paper decided that the plain negative binomial is enough for
fRNA-seq. [Choosing a distribution](../choosing-a-distribution.md) is the
page to read first.

::: preffect.zinb_fit
