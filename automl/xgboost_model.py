try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

from utils.device_utils import build_xgboost_runtime_params
from utils.train_utils import build_missing_dependency_result, run_grid_search

import config


def train_model(features, target):
    """Train XGBoost with GridSearchCV."""
    # 如果环境没装 xgboost，就先跳过并保留结构给后续安装。
    if XGBClassifier is None:
        return build_missing_dependency_result(
            model_name="xgboost_model",
            reason="xgboost is not installed.",
        )

    runtime_params = build_xgboost_runtime_params()

    # XGBoost 本体显式设成单线程，避免和外层参数搜索重复抢 CPU。
    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        **runtime_params,
    )
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    return run_grid_search(
        model_name="xgboost_model",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
