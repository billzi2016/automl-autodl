from __future__ import annotations

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:  # pragma: no cover
    TabNetClassifier = None

import config
from utils.train_utils import (
    build_completed_result,
    build_missing_dependency_result,
    evaluate_binary_metrics,
    get_torch_device,
    split_tabular_arrays,
)


def train_model(features, target):
    """Train a TabNet classifier on dense tabular features."""
    if TabNetClassifier is None:
        return build_missing_dependency_result("tabnet", "pytorch-tabnet is not installed.")

    train_features, valid_features, train_target, valid_target = split_tabular_arrays(
        features,
        target,
    )
    device = get_torch_device()
    device_name = "cuda" if str(device) == "cuda" else "cpu"

    # TabNet 在这个项目里只对 CUDA 显式开启 GPU，其他环境退回 CPU。
    estimator = TabNetClassifier(
        seed=config.RANDOM_STATE,
        verbose=0,
        device_name=device_name,
        optimizer_params={"lr": config.AUTODL_LEARNING_RATE},
    )
    estimator.fit(
        train_features,
        train_target.astype(int),
        eval_set=[(valid_features, valid_target.astype(int))],
        eval_name=["valid"],
        eval_metric=["auc"],
        max_epochs=config.AUTODL_EPOCHS,
        batch_size=config.AUTODL_BATCH_SIZE,
        virtual_batch_size=min(128, config.AUTODL_BATCH_SIZE),
    )
    probability = estimator.predict_proba(valid_features)[:, 1]
    metrics = evaluate_binary_metrics(valid_target.astype(int), probability)

    return build_completed_result(
        model_name="tabnet",
        metrics=metrics,
        best_params={
            "batch_size": config.AUTODL_BATCH_SIZE,
            "epochs": config.AUTODL_EPOCHS,
            "learning_rate": config.AUTODL_LEARNING_RATE,
        },
        extras={"device": device_name},
    )
