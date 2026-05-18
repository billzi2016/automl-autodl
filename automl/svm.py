from sklearn.svm import SVC

from utils.train_utils import run_grid_search

import config


def train_model(features, target):
    """Train support vector machine with GridSearchCV."""
    # SVM 是经典的核方法模型，这里保留线性核和 RBF 核。
    estimator = SVC(
        random_state=config.RANDOM_STATE,
    )
    param_grid = {
        "C": [0.5, 1.0, 2.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"],
        "class_weight": [None, "balanced"],
    }
    return run_grid_search(
        model_name="svm",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
