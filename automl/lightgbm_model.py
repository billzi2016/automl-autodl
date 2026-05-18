try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

from utils.device_utils import build_lightgbm_runtime_params
from utils.train_utils import build_missing_dependency_result, run_grid_search

import config


def train_model(features, target):
    """Train LightGBM with GridSearchCV."""
    # LightGBM 是工业界最常见的 boosting 关键词之一。
    if LGBMClassifier is None:
        return build_missing_dependency_result(
            model_name="lightgbm_model",
            reason="lightgbm is not installed.",
        )

    runtime_params = build_lightgbm_runtime_params()

    # LightGBM 同样固定成单线程，把并行资源让给 GridSearchCV。
    estimator = LGBMClassifier(
        objective="binary",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
        **runtime_params,
    )
    param_grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.03, 0.1],
        "max_depth": [-1, 8, 12],
        "num_leaves": [31, 63],
        "subsample": [0.8, 1.0],
    }
    return run_grid_search(
        model_name="lightgbm_model",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
