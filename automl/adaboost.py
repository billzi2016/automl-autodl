from sklearn.ensemble import AdaBoostClassifier

from utils.train_utils import run_grid_search

import config


def train_model(features, target):
    """Train AdaBoost with GridSearchCV."""
    # AdaBoost 是 boosting 家族里最经典的入门模型之一。
    estimator = AdaBoostClassifier(
        random_state=config.RANDOM_STATE,
    )
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 1.0],
    }
    return run_grid_search(
        model_name="adaboost",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
