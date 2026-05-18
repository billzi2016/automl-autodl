try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

from automl.common import build_missing_dependency_result, run_grid_search

import config


def train_model(features, target):
    """Train CatBoost with GridSearchCV."""
    # CatBoost 对类别特征场景比较常见，适合作为表格建模方案补充。
    if CatBoostClassifier is None:
        return build_missing_dependency_result(
            model_name="catboost_model",
            reason="catboost is not installed.",
        )

    # CatBoost 线程数也压成 1，避免和外层搜索同时并行。
    estimator = CatBoostClassifier(
        random_seed=config.RANDOM_STATE,
        verbose=0,
        thread_count=1,
        loss_function="Logloss",
    )
    param_grid = {
        "iterations": [200, 400],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "l2_leaf_reg": [3, 5, 7],
    }
    return run_grid_search(
        model_name="catboost_model",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
