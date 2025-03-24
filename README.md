# Analyzing Locality in XAI Methods for Tabular Data

## Project Overview
This research investigates the locality characteristics of different XAI (eXplainable AI) methods on tabular data. We focus on comparing LIME and Integrated Gradients across various models and datasets.

## Experimental Setup

### Dimensions of Analysis
- **XAI Methods**: LIME, Integrated Gradients
- **Models**: Deep learning and gradient boosting approaches
- **Datasets**: 3 standard + 3 synthetic datasets
- **Distance Measures**: 
    - lime: euclidean, manhattan, canberra, cosine, 
    - gradient: all of the above + infinity
- **Kernel Widths**: half, double, default (LIME only)

### Current Progress Matrix

#### Completed Experiments
| XAI Method           | Completed Models | Status |
|---------------------|-----------------|---------|
| LIME |  ✅ All models (Gbt, DL) | ✅ All lime-compatible distances & kernel widths |
| Integrated Gradients |  ✅ All DL-models| ✅ All distances |

| XAI Method | Model Type | Configuration | Total |
|------------|------------|---------------|--------|
| LIME (GBT) | 3 models × 6 datasets × 4 distances × 3 kernel widths |  216 experiments | ✅ Done |
| LIME (Deep) | 6 models × 6 datasets × 4 distances × 3 kernel widths |  432 experiments | ✅ Done |
| IG (Deep) | 6 models × 6 datasets × 5 distances | 180 experiments |  ✅ Done |

#### Pending Experiments
| XAI Method | Pending Models |
|------------|---------------|
| Anchor | All models (Gbt, DL) |
| Smooth Grad | All DL-models |
| ..?.. | All models |


### Datasets

#### Standard Datasets
| Dataset | Features | Samples | Description |
|---------|----------|---------|-------------|
| Higgs | 28 | 940,160 | Binary classification of Higgs boson signals |
| Jannis | 54 |57,580 | Binary classification benchmark dataset |
| MiniBooNE | 50 |72,998 | Particle identification |

To be extended to all datasets integrated into **pytorch frame**, see description here:
- [Yandex datasets](https://pytorch-frame.readthedocs.io/en/latest/generated/torch_frame.datasets.Yandex.html)
- [TabularBenchmark](https://pytorch-frame.readthedocs.io/en/latest/generated/torch_frame.datasets.TabularBenchmark.html#torch_frame.datasets.TabularBenchmark)

#### Synthetic Datasets
Using sklearns method: ```sklearn.datasets.make_classification```
[Link to dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification)

| Complexity | Features | Informative Features | Clusters per Class | 
|------------|----------|---------------------|-------------|
| Simple     | 50 | 2 | 2       
| Medium     | 50 | 10 | 3     
| Complex    | 100 | 50 |3    

### Models

#### Deep Learning Models (PyTorch-Frame)
- [ResNet (Gorishniy et al., 2021)](https://github.com/yandex-research/rtdl-revisiting-models)
- [ExcelFormer (Chen et al., 2023a)](https://github.com/WhatAShot/ExcelFormer)
- [Trompt (Chen, et al., 22023)](https://arxiv.org/abs/2305.18446)
- [FTTransformer (Gorishniy et al., 2021)](https://github.com/yandex-research/rtdl-revisiting-models)
- [TabNet (Arik Sercan O., 2021)](https://github.com/dreamquark-ai/tabnet)
- [TabTransformer (Huang et al., 2020)](https://github.com/lucidrains/tab-transformer-pytorch)
- Simple MLP

#### Statistical ML Models (PyTorch)
- LogisticRegression, implemented with pytorch for differentiability

#### Gradient Boosting Models
- XGBoost
- LightGBM

### Configuration Parameters
- **Distance Measures**: euclidean, manhattan, canberra, cosine, infinity
- **Kernel Widths** (LIME only): half of the default, double of the default, default

## Attribution
This repository contains code adapted from the python package [PyTorch Frame (PyG-team)](https://github.com/pyg-team/pytorch_geometric).  
- Original source: [GitHub link to original script](https://github.com/pyg-team/pytorch-frame/benchmark/data_frame_benchmark.py)  
- License: MIT ([link](https://github.com/pyg-team/pytorch_geometric/pytorch-frame/LICENSE))  
Modifications include dataset adaptation for our specific use case.
