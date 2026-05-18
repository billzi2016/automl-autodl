from sklearn.linear_model import LogisticRegression

from utils.train_utils import run_grid_search

import config


def train_model(features, target):
    """Train logistic regression with GridSearchCV."""
    # 逻辑回归是二分类项目里最基础也最常被问到的线性模型。
    estimator = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=config.RANDOM_STATE,
    )
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    }
    return run_grid_search(
        model_name="logistic_regression",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
