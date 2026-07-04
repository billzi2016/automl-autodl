# AutoML overview

The AutoML entrypoint is `main.py`. It prepares the data, iterates over `config.AUTOML_MODELS`, runs each model, prints a summary, and saves JSON output.

Current models:

```text
logistic_regression
svm
knn
naive_bayes
decision_tree
random_forest
extra_trees
gradient_boosting
hist_gradient_boosting
adaboost
xgboost_model
lightgbm_model
catboost_model
```

Each model file exposes `train_model(features, target)`.

## Shared result shape

Completed model:

```json
{
  "model_name": "random_forest",
  "status": "completed",
  "best_score": 0.0,
  "best_params": {},
  "cv_metrics": {
    "f1": 0.0,
    "roc_auc": 0.0,
    "accuracy": 0.0
  }
}
```

Missing optional dependency:

```json
{
  "model_name": "xgboost_model",
  "status": "skipped",
  "best_score": null,
  "best_params": {},
  "cv_metrics": {},
  "reason": "xgboost is not installed."
}
```

## Metrics

The project records F1, ROC AUC, and Accuracy. It refits and selects parameters by ROC AUC.

Accuracy can be misleading on imbalanced credit-risk data. AUC is better for ranking ability, while F1 shows thresholded classification behavior.
