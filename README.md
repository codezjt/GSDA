<div align="center">

## Gradual Migration and Style Consistency for Unsupervised Domain Adaptation

[![ICME 2023](https://img.shields.io/badge/ICME-2023-blue.svg)](#)
[![Task](https://img.shields.io/badge/Task-Unsupervised%20Domain%20Adaptation-green.svg)](#)
[![Backbone](https://img.shields.io/badge/Backbone-ResNet--50-orange.svg)](#)

</div>

> **Abstract:** *Unsupervised Domain Adaptation (UDA) aims to learn domain-invariant characteristics so that classifiers learned from the source domain can be applied to the unlabeled target domain. Existing methods often construct an intermediate domain to alleviate domain discrepancy. However, since target samples are unlabeled, a fixed and nearly equal source/target mixing ratio can confuse the model. Moreover, the style bias of CNNs is often ignored. This paper proposes to gradually reduce the distribution discrepancy between domains by constructing continuous multiple intermediate domains. To address style discrepancy, Self-Exchange and Cross-Exchange modules are proposed to reduce the style discrepancy between domains. In addition, sample probability divergence is used to select reliable pseudo-labeled samples. Experiments show significant performance improvement on several cross-domain benchmarks.*

## Introduction

This paper studies Unsupervised Domain Adaptation (UDA), where a model is trained on a labeled source domain and adapted to an unlabeled target domain. The proposed method focuses on three aspects:

1. Constructing continuous intermediate domains with a gradually adjusted mixing ratio.
2. Reducing style discrepancy between domains via feature-level style exchange.
3. Selecting reliable pseudo-labeled target samples by probability divergence.

## Method Highlights

- **Gradual Migration:** Build multiple intermediate domains between source and target domains, and dynamically adjust the mixing ratio during training.
- **Adaptive Mixing Ratio:** Use an entropy-maximizing strategy to determine a more difficult and informative mixing ratio.
- **Self-Exchange (SE):** Exchange styles between intermediate-domain samples to improve style diversity.
- **Cross-Exchange (CE):** Transfer intermediate-domain style statistics to source-domain samples to reduce style bias.
- **Probability Divergence for Pseudo Labels:** Select reliable target samples by considering the whole prediction probability distribution rather than only the maximum probability.

## Requirements

- Linux
- Python >= 3.7
- PyTorch
- CUDA (version supported by the installed PyTorch)
- Standard deep learning libraries for ResNet-50 experiments

## Getting Started

```bash
python main.py \
  --gpu 0 \
  --source amazon \
  --target dslr \
  --db_path $DATASET_PATH \
  --save_path $SAVE_PATH
```

Datasets used in the paper include:

- Office-31
- Office-Home
- ImageCLEF-DA
- DomainNet
