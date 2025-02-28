# Master Thesis: How local are local explanations?
## Currently supported Explanation Methods:
- LIME
- Gradient-based Methods:
    - Integrated gradients (IG)
    - Smooth Grad (SG + IG, SG + Vanilla Gradient)

Still to come:
- Anchors
## Currently supported models:
1. **Deep Tabular Models, implemented by Pytorch-Frame:**
- [ResNet (Gorishniy et al., 2021)](https://github.com/gorishniy/resnet)
- [ExcelFormer (Chen et al., 2023a)](https://github.com/Chen-ExcelFormer/excelformer)
- [FTTransformer (Gorishniy et al., 2021)](https://github.com/gorishniy/fttransformer)
- [TabNet (Arik Sercan O., 2021)](https://github.com/dreamquark-ai/tabnet)
- [TabTransformer (Huang et al., 2020)](https://github.com/hyungkwonko/tab-transformer)
- Simple MLP


2. **Gradient Boosting Models** (non-gradient based XAI methods only)

3. **"TabInception"**, i.e. Pretrained last layer of Inception Net for Feature Vector Classification

### Implemented Model classes:
- LightGBM
- ExcelFormer
- MLP
- TabInception

## Currently supported Datasets:
All Datasets integrated into pytorch frame:
e.g.:
- [Yandex datasets](https://pytorch-frame.readthedocs.io/en/latest/generated/torch_frame.datasets.Yandex.html)
- [TabularBenchmark](https://pytorch-frame.readthedocs.io/en/latest/generated/torch_frame.datasets.TabularBenchmark.html#torch_frame.datasets.TabularBenchmark)


### We trained and tuned models on the following datasets

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

**ImageNet**
- Multiclass (1000) problem
- For Lime tabular: Preprocessing to feature vectors using Inception Net v3.

**Cats vs Dogs**
[Link to dataset](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.kaggle.com/competitions/dogs-vs-cats&ved=2ahUKEwj5lJSc_6mLAxVs3AIHHbdyJKIQFnoECAgQAQ&usg=AOvVaw0KmiBfaaItAQMS2Ti6aZ0H)
- Binary Target
- Images of Cats and dogs


## Attribution
This repository contains code adapted from the python package [PyTorch Frame (PyG-team)](https://github.com/pyg-team/pytorch_geometric).  
- Original source: [GitHub link to original script](https://github.com/pyg-team/pytorch-frame/benchmark/data_frame_benchmark.py)  
- License: MIT ([link](https://github.com/pyg-team/pytorch_geometric/pytorch-frame/LICENSE))  
Modifications include dataset adaptation for our specific use case.
