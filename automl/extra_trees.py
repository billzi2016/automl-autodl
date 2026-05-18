from sklearn.ensemble import ExtraTreesClassifier

from utils.train_utils import run_grid_search

import config


def train_model(features, target):
    """Train extra trees with GridSearchCV."""
    # Extra Trees 常被用来和随机森林对比泛化能力。
    # 这里也固定单线程，避免双重并行把机器打满。
    estimator = ExtraTreesClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 16, None],
        "min_samples_split": [2, 10],
        "class_weight": [None, "balanced"],
    }
    return run_grid_search(
        model_name="extra_trees",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
