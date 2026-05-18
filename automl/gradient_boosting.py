from sklearn.ensemble import GradientBoostingClassifier

from automl.common import run_grid_search

import config


def train_model(features, target):
    """Train gradient boosting with GridSearchCV."""
    # 传统 GBDT 适合用于展示 boosting 的基本思路。
    estimator = GradientBoostingClassifier(
        random_state=config.RANDOM_STATE,
    )
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    }
    return run_grid_search(
        model_name="gradient_boosting",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
