# GridSearchCV 详解

`GridSearchCV` 的公共封装在 `utils/train_utils.py`：

```python
def run_grid_search(model_name, estimator, param_grid, features, target):
    ...
```

每个 AutoML 模型只需要提供两样东西：

- `estimator`
- `param_grid`

剩下的交叉验证、指标、refit 和结果整理都走统一逻辑。

## 交叉验证

当前使用：

```python
StratifiedKFold(
    n_splits=config.CV_FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE,
)
```

`config.CV_FOLDS = 3`。

`StratifiedKFold` 会尽量保持每一折里的正负样本比例接近原数据。对 `TARGET` 这种二分类标签来说，这比普通 KFold 更稳。

## 多指标评分

当前评分配置：

```python
SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}
```

`GridSearchCV` 会对每组参数都计算三类指标。结果里取最佳参数对应的：

- `mean_test_f1`
- `mean_test_roc_auc`
- `mean_test_accuracy`

## Refit 规则

当前配置：

```python
REFIT_METRIC = "roc_auc"
```

这表示搜索结束后，最佳参数按 AUC 选择。`best_params_` 不是 F1 最高的一组，也不是 Accuracy 最高的一组，而是 `mean_test_roc_auc` 最高的一组。

## 参数搜索规模

网格搜索会穷举每个参数组合。比如 Random Forest 的网格：

```python
{
    "n_estimators": [200, 400],
    "max_depth": [8, 16, None],
    "min_samples_split": [2, 10],
    "class_weight": [None, "balanced_subsample"],
}
```

组合数是 `2 * 3 * 2 * 2 = 24`。3 折交叉验证后，实际训练次数是 `24 * 3 = 72`。

这也是为什么项目把很多模型内部线程数设为 1。外层已经在并行跑组合，内层再并行通常不划算。

## 输出整理

搜索完成后会取：

```python
best_index = search.best_index_
```

然后从 `search.cv_results_` 里读最佳参数对应的三个平均测试指标。最终用 `build_completed_result()` 生成统一结构。

这个项目没有保存每一组参数的完整 `cv_results_`。当前输出偏向汇总比较，不适合直接做参数搜索过程分析。如果后续要画参数曲线，可以在 `run_grid_search()` 里额外保存完整结果。
