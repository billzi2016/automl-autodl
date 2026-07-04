# GridSearchCV details

The shared wrapper is `run_grid_search()` in `utils/train_utils.py`.

Each AutoML model provides:

- an estimator
- a parameter grid

The wrapper handles cross validation, scoring, refit, and result formatting.

## Cross validation

The project uses:

```python
StratifiedKFold(
    n_splits=config.CV_FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE,
)
```

`config.CV_FOLDS` is currently `3`.

`StratifiedKFold` keeps the class ratio similar across folds, which is useful for binary risk labels.

## Scoring

```python
SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}
```

The best parameter set is selected with:

```python
REFIT_METRIC = "roc_auc"
```

So `best_params_` means “best by mean validation ROC AUC”, not best by F1 or Accuracy.

## Search cost

Grid search tries every parameter combination. A Random Forest grid with 24 combinations and 3 folds runs 72 model fits.

Several estimators are configured with `n_jobs=1` internally because the outer `GridSearchCV` already parallelizes with `n_jobs=-1`.
