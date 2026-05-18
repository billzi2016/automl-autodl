from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config


def run_grid_search(
    model_name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    features: Any,
    target: Any,
) -> dict[str, Any]:
    """Run GridSearchCV with a shared configuration."""
    # 统一在这里维护交叉验证和并行策略，保证所有模型行为一致。
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    # 同时保留 F1、AUC 和 Accuracy，最后仍按 AUC 选择最优参数。
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=config.SCORING,
        refit=config.REFIT_METRIC,
        cv=cv,
        n_jobs=config.GRIDSEARCH_N_JOBS,
        verbose=1,
    )
    search.fit(features, target)

    best_index = search.best_index_
    cv_metrics = {
        "f1": float(search.cv_results_["mean_test_f1"][best_index]),
        "roc_auc": float(search.cv_results_["mean_test_roc_auc"][best_index]),
        "accuracy": float(search.cv_results_["mean_test_accuracy"][best_index]),
    }

    return {
        "model_name": model_name,
        "status": "completed",
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_metrics": cv_metrics,
    }


def build_missing_dependency_result(model_name: str, reason: str) -> dict[str, Any]:
    """Return a structured result for models that cannot run yet."""
    return {
        "model_name": model_name,
        "status": "skipped",
        "best_score": None,
        "best_params": {},
        "cv_metrics": {},
        "reason": reason,
    }
