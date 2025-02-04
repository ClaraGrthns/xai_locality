# Master Thesis: How local are local explanations?
## Currently supported explanation methods and models:

### LIME

| Models ↓, Dataset →| Jannis | SynData: 2 inf feat., 2 cluster  | SynData: 20 inf feat. 2 cluster  | SynData: 20 inf feat., 5 cluster  | Higgs | Feature Vectors of ImageNet | Feature Vectors of Cats vs. Dogs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Light GBM | x | x | x | x | (x) |  |  |
| Inception V3 |  |  |  |  |  | x |  |

### Gradient Methods

**Integrated Gradients **

|  Models ↓ Dataset →, | ImageNet | Cats vs. Dogs |
| --- | --- | --- |
| TabPFN |  |  |
| Inception V3 |  | x |

**SmoothGrad + Integrated Gradients**

|  Models ↓, Dataset →| ImageNet | Cats vs. Dogs |
| --- | --- | --- |
| TabPFN |  |  |
| Inception V3 |  | x |




### Currently supported Datasets:
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



