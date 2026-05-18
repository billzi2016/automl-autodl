from sklearn.ensemble import RandomForestClassifier

from utils.train_utils import run_grid_search

import config


def train_model(features, target):
    """Train random forest with GridSearchCV."""
    # 随机森林是树模型里的经典强基线。
    # 模型本体限制单线程，把并行资源留给外层 GridSearchCV。
    estimator = RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 16, None],
        "min_samples_split": [2, 10],
        "class_weight": [None, "balanced_subsample"],
    }
    return run_grid_search(
        model_name="random_forest",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
