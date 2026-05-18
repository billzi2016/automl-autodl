from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

import config
from utils.train_utils import build_missing_dependency_result, train_torch_binary_classifier


BaseModule = nn.Module if nn is not None else object


class MLPClassifier(BaseModule):
    """Simple multilayer perceptron for dense tabular inputs."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.AUTODL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.AUTODL_DROPOUT),
            nn.Linear(config.AUTODL_HIDDEN_DIM, config.AUTODL_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.AUTODL_DROPOUT),
            nn.Linear(config.AUTODL_HIDDEN_DIM // 2, 1),
        )

    def forward(self, features):
        return self.network(features)


def train_model(features, target):
    """Train an MLP classifier on tabular features."""
    if nn is None:
        return build_missing_dependency_result("mlp", "torch is not installed.")

    model = MLPClassifier(input_dim=features.shape[1])
    return train_torch_binary_classifier("mlp", model, features, target)
