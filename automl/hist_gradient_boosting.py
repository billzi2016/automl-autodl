from sklearn.ensemble import HistGradientBoostingClassifier

from utils.train_utils import run_grid_search

import config


def train_model(features, target):
    """Train histogram gradient boosting with GridSearchCV."""
    # Hist GBDT 是 sklearn 里更现代的 boosting 实现。
    estimator = HistGradientBoostingClassifier(
        random_state=config.RANDOM_STATE,
    )
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 8, None],
        "max_leaf_nodes": [15, 31],
        "min_samples_leaf": [20, 50],
    }
    return run_grid_search(
        model_name="hist_gradient_boosting",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
