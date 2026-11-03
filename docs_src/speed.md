# Speed, version 0.1 against version 0.2

Measured 2026-10-16. Forward pass, backward pass and one optimiser step, on
500 samples by 2,000 genes, with one correction variable. **Median of three
runs**, 20 steps each, on a machine whose load average was between 130 and 180
throughout, with the two versions run interleaved so they met the same
conditions.

Reproduce with `tools/benchmark_v01.py` and `tools/benchmark_v02.py`.

**The two versions ran on different PyTorch versions**, 0.1 on torch 2.5.1 and
0.2 on torch 2.14.1, because 0.1's environment is Python 3.9 and 0.2 needs
torch 2.6 or later. This page did not say so until 2026-10-27. The
same-version comparison is below, and it changes the conclusion.

The first version of this page left the network reconstruction out of the
stand-in objective, so the graph models' network decoder was never exercised.
That is the one place the two versions differ structurally, so the numbers
below include it and the earlier ones understated version 0.2.

## CPU, eight threads

| model | 0.1 | 0.2 | |
|---|---|---|---|
| simple | 19.7 ms | 23.6 ms | 0.2 is 1.20x slower |
| single | 92.3 ms | 113.1 ms | 0.2 is 1.23x slower |
| full | 167.9 ms | 202.7 ms | 0.2 is 1.21x slower |

## One A100 GPU

| model | 0.1 | 0.2 | |
|---|---|---|---|
| simple | 6.3 ms | 4.6 ms | **0.2 is 1.4x faster** |
| single | 14.1 ms | 16.6 ms | 0.2 is 1.18x slower |
| full | 23.0 ms | 23.0 ms | the same |

## Where version 0.2 is genuinely faster

The sample-sample network decoder, and by a wide margin. Version 0.1
reconstructs the network with a linear layer whose output width **is the batch
size**; version 0.2 takes an inner product of the latent with itself and has no
parameters there at all. At 500 samples, one full step of that component alone:

| | time | parameters |
|---|---|---|
| 0.1, `Linear(128 -> 500)` then sigmoid | 1035 us | 64,500 |
| 0.2, `sigmoid(z @ z.T)` | 294 us | 0 |

**3.5x faster, with nothing to train.** It also stops the architecture changing
when the batch size changes, which was finding 20.

Including this component narrowed the overall CPU gap from about 1.35x to
about 1.21x, which is the size of win it is worth: real, but a small part of a
step dominated by the gene-wide matmuls both versions share.

## What this said, and what was wrong with it

The first version of this section said version 0.2 was about a third slower
than version 0.1 on a CPU, that the rewrite had cost roughly 35% of the speed,
and that why had not been investigated. It stood for eleven days. It compared
version 0.1 on torch 2.5.1 with version 0.2 on torch 2.14.1, and the
difference it measured is the PyTorch version's, not the rewrite's. Finding
46.

## Measured on the same PyTorch, 2026-10-27

Version 0.2's benchmark run under torch 2.5.1, in a throwaway environment with
the package's torch floor bypassed, interleaved with version 0.1 and with
version 0.2 on torch 2.14.1 in the same minute. Simple model, 500 samples by
2,000 genes, load average about 130, medians of three:

| threads | 0.1 on torch 2.5.1 | 0.2 on torch 2.5.1 | 0.2 on torch 2.14.1 |
|---|---|---|---|
| 8 | 16.4 ms | **13.1 ms** | 19.5 ms |
| 1 | 44.5 ms | 45.8 ms | 47.8 ms |

**On the same PyTorch the rewrite is level or faster**, which is what 6.7x
fewer parameters would predict. On torch 2.14.1 it is slower, and only when
there are threads to use. One component is identified: `nn.Dropout` on a
hidden-sized tensor costs 0.07 to 0.11 ms under torch 2.5.1 and 0.65 to 0.85
ms under 2.14.1, seven times slower, and a random-number kernel does not
parallelise; three dropouts a step is about 2 ms of the 6 ms gap. The rest is
issue #24. On a GPU the two versions are within ten per cent of each other,
which is to say indistinguishable at this precision.

`tools/profile_step.py` runs either version's step under the profiler, and its
`--control` times plain layers of the same shapes with no PREFFECT code in
them, which is how the PyTorch version was separated from the model.

## Where it entered, and what to do

A sweep of the CPU wheel of every release from 2.6.0 to 2.14.1 puts the change
between 2.6.0 and 2.7.1: a dropout mask on 500 by 128 costs 0.09 ms at 2.6.0
and 0.55 to 0.83 ms at every release after. A per-operator profile of 0.2's
step says dropout is a third of the 4 ms difference, `zero_grad` a tenth, and
the rest a 10 to 40 per cent increase across ordinary kernels. The tables are in
the development record, the decision of 2026-10-27 on installing PyTorch 2.6
on a CPU.

**On a machine without a GPU, install PyTorch 2.6 before PREFFECT.** It is the
package's floor, so nothing is lost, and 0.2's step runs at 16.1 ms on it
against 20.1 ms on 2.14.1 under the same conditions. [Getting
started](getting-started.md) gives the command.

| model | 0.1 parameters | 0.2 parameters | |
|---|---|---|---|
| simple | 5,350,973 | 800,414 | 0.2 is 6.7x smaller |
| single | 9,956,350 | 9,058,496 | 0.9x |
| full | 19,277,480 | 17,598,484 | 0.9x |

The simple model's difference is finding 8: version 0.1 allocates layers the
chosen model never uses, including a graph attention layer of several million
parameters inside a model with no graph. Those layers cost memory and
initialisation time but are never executed, which is why removing them shrinks
the model without making it faster.

## The GPU is worth using

For version 0.2, one A100 against eight CPU threads:

| model | speed-up |
|---|---|
| simple | 5.0x |
| single | 8.8x |
| full | 7.0x |

The graph models gain most, which is what you would expect from attention.

## What this measurement is not

It times the computation, not a complete run. Version 0.1's session machinery
cannot be driven without writing files, so comparing whole runs would compare
file handling as much as arithmetic. It also uses a stand-in objective, the sum
of the fitted means, because version 0.1's loss needs its session state. Both
versions get the same treatment, so the comparison is fair, but neither number
is "how long my analysis takes".

Thread count matters separately and is covered in
[getting started](getting-started.md).
