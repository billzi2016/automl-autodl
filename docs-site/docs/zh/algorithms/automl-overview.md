# AutoML 总览

AutoML 入口是 `main.py`。它不直接写模型细节，只负责调度：

```python
features, target = prepare_training_data()

for model_name in config.AUTOML_MODELS:
    runner = MODEL_RUNNERS[model_name]
    result = runner(features, target)
    results.append(result)
```

模型实现都在 `automl/` 下，每个文件暴露一个 `train_model(features, target)`。

## 模型列表

`config.AUTOML_MODELS` 当前包含：

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

## 统一搜索入口

每个模型最终都会调用 `utils.train_utils.run_grid_search()`，除了缺少依赖时会返回 `skipped` 的 XGBoost、LightGBM、CatBoost。

返回结构统一：

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

如果依赖缺失：

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

## 指标

`config.SCORING` 同时记录：

- `f1`
- `roc_auc`
- `accuracy`

最终用 `roc_auc` 做 `refit`，所以 `best_score` 对应最佳参数下的 AUC。

在风控二分类任务里，Accuracy 往往会被类别比例影响。AUC 更适合看模型对正负样本的排序能力，F1 则可以补充观察阈值为 0.5 时的分类表现。

## 并行策略

`GridSearchCV` 使用 `n_jobs=-1`。几个模型内部设置为单线程，例如 Random Forest、Extra Trees、XGBoost、LightGBM、CatBoost。原因是外层参数搜索已经并行，如果模型内部也开满线程，机器会被双重并行拖慢。
