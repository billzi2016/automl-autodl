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


class BLSTMClassifier(BaseModule):
    """Bidirectional LSTM over reshaped tabular feature sequences."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config.AUTODL_HIDDEN_DIM // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.AUTODL_DROPOUT,
        )
        self.head = nn.Sequential(
            nn.Linear(config.AUTODL_HIDDEN_DIM, config.AUTODL_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.AUTODL_DROPOUT),
            nn.Linear(config.AUTODL_HIDDEN_DIM // 2, 1),
        )
        self.input_dim = input_dim

    def forward(self, features):
        sequence = features.unsqueeze(-1)
        output, _ = self.lstm(sequence)
        pooled = output[:, -1, :]
        return self.head(pooled)


def train_model(features, target):
    """Train a bidirectional LSTM classifier."""
    if nn is None:
        return build_missing_dependency_result("blstm", "torch is not installed.")

    model = BLSTMClassifier(input_dim=features.shape[1])
    return train_torch_binary_classifier("blstm", model, features, target)
