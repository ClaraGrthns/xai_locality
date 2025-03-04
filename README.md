# Master Thesis: How local are local explanations?
## Current Status
We trained the following models on these datasets and run the locality analysis: 

| XAI Method           | Synthetic Data (Simple) | Synthetic Data (Medium) | Synthetic Data (Complex) | Jannis | MiniBooNE | Higgs |
|----------------------|---------------------|---------------------|----------------------|--------|----|-------|
| LIME                 | LightGBM, MLP, ExcelFormer, Trompt | LightGBM, MLP, ExcelFormer, Trompt | LightGBM, MLP, ExcelFormer, Trompt | LightGBM, MLP, ExcelFormer, Trompt | LightGBM, MLP, ExcelFormer, Trompt | LightGBM, MLP, ExcelFormer, Trompt |
| Integrated Gradients |  MLP, ExcelFormer, Trompt |  MLP, ExcelFormer, Trompt |  MLP, ExcelFormer, Trompt |  MLP, ExcelFormer, Trompt |  MLP, ExcelFormer, Trompt | MLP, ExcelFormer, Trompt |

## Currently supported Explanation Methods:
- LIME
- Gradient-based Methods:
    - Integrated gradients (IG)
    - Smooth Grad (SG + IG, SG + Vanilla Gradient)

Still to come:
- Anchors
## Currently supported models:
(but not necessarily trained and finetuned on the respective datasets)
1. **Deep Tabular Models, implemented by Pytorch-Frame:**
- [ResNet (Gorishniy et al., 2021)](https://github.com/yandex-research/rtdl-revisiting-models)
- [ExcelFormer (Chen et al., 2023a)](https://github.com/WhatAShot/ExcelFormer)
- [Trompt (Chen, et al., 22023)](https://arxiv.org/abs/2305.18446)
- [FTTransformer (Gorishniy et al., 2021)](https://github.com/yandex-research/rtdl-revisiting-models)
- [TabNet (Arik Sercan O., 2021)](https://github.com/dreamquark-ai/tabnet)
- [TabTransformer (Huang et al., 2020)](https://github.com/lucidrains/tab-transformer-pytorch)
- Simple MLP


2. **Gradient Boosting Models** (For non-gradient based XAI methods only)

3. **"TabInception"**, i.e. Pretrained last layer of Inception Net for Feature Vector Classification


## Currently supported Datasets:
All Datasets integrated into **pytorch frame**:
e.g.:
- [Yandex datasets](https://pytorch-frame.readthedocs.io/en/latest/generated/torch_frame.datasets.Yandex.html)
- [TabularBenchmark](https://pytorch-frame.readthedocs.io/en/latest/generated/torch_frame.datasets.TabularBenchmark.html#torch_frame.datasets.TabularBenchmark)

Other Datasets, i.e. **custom** datasets
- Feature vectors of Inception Net, finetuned on binary classification of the datasets [Cats vs. Dogs](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.kaggle.com/competitions/dogs-vs-cats&ved=2ahUKEwj5lJSc_6mLAxVs3AIHHbdyJKIQFnoECAgQAQ&usg=AOvVaw0KmiBfaaItAQMS2Ti6aZ0H)
- Feature vectors of Inception Net, finetuned on binary classification of the datasets [ImageNet](https://www.image-net.org)
- [Synthetic Data Generation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) with sklearn 


### Deatils on the datasets

**Higgs**:

[Link to dataset](https://archive.ics.uci.edu/dataset/280/higgs)

- Binary Target: a signal process which produces Higgs bosons and a background process which does not.
- 28 continuous features
- 11000000 sampels
- Description: 
The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes.

**Jannis**:
[Link to dataset](https://www.openml.org/search?type=data&id=41168&sort=runs&status=active)

- Binary Target.
- 54 continouts features
- Description: Dataset used in the tabular data benchmark https://github.com/LeoGrin/tabular-benchmark, transformed in the same way. This dataset belongs to the "classification on numerical features" benchmark. Original description:

**Synthetic Data**:
Using sklearns method: ```sklearn.datasets.make_classification```
[Link to dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification)
- Binary Target
- 50 continuous features
- Varying informative and redundant features
- Varying clusters per class
- class separation: 0.9
- Description: Generate a random n-class classification problem.
This initially creates clusters of points normally distributed (std=1) about vertices of an ```n_informative```-dimensional hypercube with sides of length ```2*class_sep ```and assigns an equal number of clusters to each class. It introduces interdependence between these features and adds various types of further noise to the data. Without shuffling, ```X``` horizontally stacks features in the following order: the primary ```n_informative``` features, followed by ```n_redundant```linear combinations of the informative features, followed by n_repeated duplicates, drawn randomly with replacement from the informative and redundant features. The remaining features are filled with random noise. Thus, without shuffling, all useful features are contained in the columns ```X[:, :n_informative + n_redundant + n_repeated]```.


## Attribution
This repository contains code adapted from the python package [PyTorch Frame (PyG-team)](https://github.com/pyg-team/pytorch_geometric).  
- Original source: [GitHub link to original script](https://github.com/pyg-team/pytorch-frame/benchmark/data_frame_benchmark.py)  
- License: MIT ([link](https://github.com/pyg-team/pytorch_geometric/pytorch-frame/LICENSE))  
Modifications include dataset adaptation for our specific use case.
