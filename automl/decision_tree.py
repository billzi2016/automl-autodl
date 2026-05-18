from sklearn.tree import DecisionTreeClassifier

from utils.train_utils import run_grid_search

import config


def train_model(features, target):
    """Train decision tree with GridSearchCV."""
    # 决策树本身可解释性强，也常作为集成模型的基础。
    estimator = DecisionTreeClassifier(
        random_state=config.RANDOM_STATE,
    )
    param_grid = {
        "max_depth": [4, 8, 12, None],
        "min_samples_split": [2, 10, 30],
        "min_samples_leaf": [1, 5, 10],
        "class_weight": [None, "balanced"],
    }
    return run_grid_search(
        model_name="decision_tree",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
