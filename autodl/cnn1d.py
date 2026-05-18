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


class CNN1DClassifier(BaseModule):
    """1D convolutional classifier over tabular feature sequences."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, config.AUTODL_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.AUTODL_DROPOUT),
            nn.Linear(config.AUTODL_HIDDEN_DIM // 2, 1),
        )
        self.input_dim = input_dim

    def forward(self, features):
        sequence = features.unsqueeze(1)
        extracted = self.feature_extractor(sequence)
        return self.head(extracted)


def train_model(features, target):
    """Train a 1D CNN classifier."""
    if nn is None:
        return build_missing_dependency_result("cnn1d", "torch is not installed.")

    model = CNN1DClassifier(input_dim=features.shape[1])
    return train_torch_binary_classifier("cnn1d", model, features, target)
