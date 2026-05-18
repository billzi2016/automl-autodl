from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

import config
from utils.device_utils import detect_compute_environment

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


def run_grid_search(
    model_name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    features: Any,
    target: Any,
) -> dict[str, Any]:
    """Run GridSearchCV with a shared configuration."""
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
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

    return build_completed_result(
        model_name=model_name,
        metrics=cv_metrics,
        best_params=search.best_params_,
    )


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


def build_completed_result(
    model_name: str,
    metrics: dict[str, float],
    best_params: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unified result payload for both AutoML and AutoDL."""
    result = {
        "model_name": model_name,
        "status": "completed",
        "best_score": float(metrics["roc_auc"]),
        "best_params": best_params,
        "cv_metrics": metrics,
    }
    if extras:
        result.update(extras)
    return result


class NumpyClassificationDataset(Dataset):
    """Simple dataset wrapper for dense tabular arrays."""

    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.features[index], self.target[index]


def get_torch_device():
    """Choose the best available device for PyTorch models."""
    if torch is None:
        return None

    environment = detect_compute_environment()
    if environment["preferred_device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if environment["preferred_device"] == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_tabular_arrays(features, target):
    """Split dense tabular data into train and validation arrays."""
    feature_array = features.to_numpy(dtype=np.float32)
    target_array = target.to_numpy(dtype=np.float32)
    return train_test_split(
        feature_array,
        target_array,
        test_size=config.AUTODL_VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=target_array,
    )


def build_data_loaders(train_features, train_target, valid_features, valid_target):
    """Build training and validation dataloaders for tabular tensors."""
    train_dataset = NumpyClassificationDataset(train_features, train_target)
    valid_dataset = NumpyClassificationDataset(valid_features, valid_target)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.AUTODL_BATCH_SIZE,
        shuffle=True,
        num_workers=config.AUTODL_NUM_WORKERS,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.AUTODL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.AUTODL_NUM_WORKERS,
    )
    return train_loader, valid_loader


def evaluate_binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute binary classification metrics from probabilities."""
    predicted_label = (probability >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(target, predicted_label)),
        "accuracy": float(accuracy_score(target, predicted_label)),
    }

    if len(np.unique(target)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(target, probability))
    else:
        metrics["roc_auc"] = 0.5

    return metrics


def train_torch_binary_classifier(model_name: str, model, features, target) -> dict[str, Any]:
    """Train a PyTorch binary classifier on dense tabular features."""
    if torch is None or nn is None:
        return build_missing_dependency_result(
            model_name=model_name,
            reason="torch is not installed.",
        )

    device = get_torch_device()
    train_features, valid_features, train_target, valid_target = split_tabular_arrays(
        features,
        target,
    )
    train_loader, valid_loader = build_data_loaders(
        train_features,
        train_target,
        valid_features,
        valid_target,
    )

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.AUTODL_LEARNING_RATE,
        weight_decay=config.AUTODL_WEIGHT_DECAY,
    )

    best_metrics = {"f1": 0.0, "roc_auc": 0.0, "accuracy": 0.0}
    best_state = copy.deepcopy(model.state_dict())
    no_improvement_epochs = 0

    for _ in range(config.AUTODL_EPOCHS):
        model.train()
        for batch_features, batch_target in train_loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device).float()

            optimizer.zero_grad()
            logits = model(batch_features).squeeze(-1)
            loss = criterion(logits, batch_target)
            loss.backward()
            optimizer.step()

        current_metrics = evaluate_torch_model(model, valid_loader, valid_target, device)
        if current_metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_metrics = current_metrics
            best_state = copy.deepcopy(model.state_dict())
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= config.AUTODL_EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)
    return build_completed_result(
        model_name=model_name,
        metrics=best_metrics,
        best_params={
            "batch_size": config.AUTODL_BATCH_SIZE,
            "epochs": config.AUTODL_EPOCHS,
            "learning_rate": config.AUTODL_LEARNING_RATE,
        },
        extras={
            "device": str(device),
        },
    )


def evaluate_torch_model(model, data_loader, valid_target, device) -> dict[str, float]:
    """Evaluate a torch model on the validation loader."""
    model.eval()
    probability_batches = []

    with torch.no_grad():
        for batch_features, _ in data_loader:
            batch_features = batch_features.to(device)
            logits = model(batch_features).squeeze(-1)
            probability = torch.sigmoid(logits).detach().cpu().numpy()
            probability_batches.append(probability)

    stacked_probability = np.concatenate(probability_batches, axis=0)
    return evaluate_binary_metrics(valid_target.astype(int), stacked_probability)
