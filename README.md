# CNN vs ViT: four experiments

Four experiments comparing CNNs and ViTs on the same data, same setup, with plots to back up every claim.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

I wanted to know how CNNs and ViTs actually differ, not how they differ on paper. Reading papers kept giving me the same answer ("transformers are better") without telling me better at what, or on how much data. So I ran the comparison myself.

Four experiments. Same data, same optimizer, same epochs. Only the architecture changes.

## Why I did this

Most comparisons I found either explained the theory or trained the two models under completely different conditions, which makes it impossible to tell whether the difference comes from the architecture or from everything else.

So I tried to control for that. Same everything except the model.

I expected the results to mostly confirm what people already say, and they did. But seeing it happen is different from reading it, which is the whole reason I bothered.

Here's the map:

| Phase | What it tests | Short version |
|---|---|---|
| 1 | Synthetic lines | CNNs learn basic spatial patterns faster |
| 2 | Shuffled MNIST | CNNs lean on local structure |
| 3 | Data efficiency | ViTs need a lot more data |
| 4 | Visualization | One looks at edges, the other at relationships |

## Phase 1: synthetic lines

16x16 grayscale images. Each one has either a vertical line or a horizontal line in it. Two classes, and a human can solve it in about half a second.

I picked it for exactly that reason. The task itself is trivial, so any gap between the two models has to be about learning and not about the problem being hard.

The CNN got to 100% test accuracy in 2 epochs. The ViT took 4.

Both end up at 100%, so if you only look at final accuracy there's no difference at all. The difference is entirely in how long it took to get there, and that's the most direct look at inductive bias I've managed to get.

A convolution already assumes nearby pixels are related, and that a feature found in one corner is probably useful in another. The ViT assumes none of that. It gets patches and has to work out from the data that position matters and that patches relate to each other. That costs it a couple of epochs.

Same destination, different amount of figuring out.

![Samples from the dataset](results/phase1_samples.png)
*Sample images from the synthetic lines dataset.*

![CNN vs ViT training curves](results/phase1_comparison.png)
*How fast each model got to 100%.*
