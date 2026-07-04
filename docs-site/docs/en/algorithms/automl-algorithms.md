# AutoML algorithms

All AutoML models consume the same preprocessed dense feature matrix and binary `TARGET` labels. They share `GridSearchCV`, 3-fold stratified cross validation, AUC/F1/Accuracy tracking, and AUC-based refit.

## Logistic Regression

Logistic Regression learns one weight per feature, combines them into a linear score, and turns that score into a probability with a sigmoid. Positive weights push the prediction toward `TARGET=1`; negative weights push it toward `TARGET=0`.

Grid:

```python
{
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
}
```

`C` controls regularization inversely. `l1` can zero out coefficients, while `l2` shrinks them smoothly. This model is the linear baseline.

## SVM

SVM tries to find a separating boundary with a wide margin. A linear kernel searches in the original feature space; an RBF kernel allows curved boundaries.

Grid:

```python
{
    "C": [0.5, 1.0, 2.0],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale"],
    "class_weight": [None, "balanced"],
}
```

SVM is scale-sensitive, so the project’s numeric standardization matters here.

## KNN

KNN stores the training examples. For a new row, it finds the closest training rows and predicts from their labels.

Grid:

```python
{
    "n_neighbors": [5, 11, 21],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
```

`p=1` is Manhattan distance; `p=2` is Euclidean distance. KNN is useful as a distance-based baseline, but high-dimensional one-hot features can make it weaker.

## Gaussian Naive Bayes

Gaussian Naive Bayes estimates what each feature looks like inside each class, then compares which class makes a row more likely. It assumes features are conditionally independent, which is rarely fully true, but the model is fast.

Grid:

```python
{
    "var_smoothing": [1e-9, 1e-8, 1e-7],
}
```

## Decision Tree

A decision tree is a chain of splits. During training, it chooses a feature and threshold that make the child nodes purer, then repeats.

Grid:

```python
{
    "max_depth": [4, 8, 12, None],
    "min_samples_split": [2, 10, 30],
    "min_samples_leaf": [1, 5, 10],
    "class_weight": [None, "balanced"],
}
```

Deep trees can memorize training data. `min_samples_leaf` and `min_samples_split` keep the tree from creating tiny, fragile leaves.

## Random Forest

Random Forest trains many decision trees on random samples and random feature subsets, then averages their predictions. It reduces the instability of a single tree.

Grid:

```python
{
    "n_estimators": [200, 400],
    "max_depth": [8, 16, None],
    "min_samples_split": [2, 10],
    "class_weight": [None, "balanced_subsample"],
}
```

The estimator uses `n_jobs=1` because the outer grid search is already parallel.

## Extra Trees

Extra Trees is similar to Random Forest, but it injects more randomness into split selection. This can reduce variance, though it may also lose useful structure.

Grid:

```python
{
    "n_estimators": [200, 400],
    "max_depth": [8, 16, None],
    "min_samples_split": [2, 10],
    "class_weight": [None, "balanced"],
}
```

Compare it with Random Forest to see whether stronger randomization helps this dataset.

## Gradient Boosting

Gradient Boosting trains trees sequentially. Each new tree tries to fix what the previous ensemble still gets wrong.

Grid:

```python
{
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0],
}
```

`learning_rate` and `n_estimators` should be read together: smaller steps usually need more trees.

## HistGradientBoosting

HistGradientBoosting is a histogram-based gradient boosting implementation. It bins continuous values before searching splits, which often trains faster on larger tabular datasets.

Grid:

```python
{
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 8, None],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],
}
```

## AdaBoost

AdaBoost increases the weight of samples that previous weak learners misclassified, so later learners focus on harder rows.

Grid:

```python
{
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 1.0],
}
```

It can be sensitive to noisy labels because it keeps emphasizing difficult examples.

## XGBoost

XGBoost is a highly engineered gradient boosting tree library. It trains trees sequentially with regularization, sampling, and efficient histogram-style training.

Grid:

```python
{
    "n_estimators": [200, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
```

CUDA sets `tree_method="hist"` and `device="cuda"`. CPU keeps `tree_method="hist"`.

## LightGBM

LightGBM is another gradient boosting tree library, commonly strong on tabular data. Its leaf-wise growth can be powerful but needs complexity control.

Grid:

```python
{
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.1],
    "max_depth": [-1, 8, 12],
    "num_leaves": [31, 63],
    "subsample": [0.8, 1.0],
}
```

`num_leaves` is one of the main complexity knobs.

## CatBoost

CatBoost is a boosting tree library with strong support for categorical data. In this project, categorical fields are already one-hot encoded before CatBoost sees them.

Grid:

```python
{
    "iterations": [200, 400],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "l2_leaf_reg": [3, 5, 7],
}
```

CUDA sets `task_type="GPU"` and `devices="0"`; otherwise it uses CPU.
