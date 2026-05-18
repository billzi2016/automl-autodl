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


class TransformerClassifier(BaseModule):
    """Transformer encoder for dense tabular features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Linear(1, config.AUTODL_SEQUENCE_EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.AUTODL_SEQUENCE_EMBED_DIM,
            nhead=config.AUTODL_NUM_HEADS,
            dim_feedforward=config.AUTODL_HIDDEN_DIM,
            dropout=config.AUTODL_DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.AUTODL_NUM_TRANSFORMER_LAYERS,
        )
        self.head = nn.Sequential(
            nn.Linear(config.AUTODL_SEQUENCE_EMBED_DIM, config.AUTODL_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.AUTODL_DROPOUT),
            nn.Linear(config.AUTODL_HIDDEN_DIM // 2, 1),
        )
        self.input_dim = input_dim

    def forward(self, features):
        sequence = features.unsqueeze(-1)
        embedded = self.embedding(sequence)
        encoded = self.encoder(embedded)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def train_model(features, target):
    """Train a transformer classifier."""
    if nn is None:
        return build_missing_dependency_result("transformer", "torch is not installed.")

    model = TransformerClassifier(input_dim=features.shape[1])
    return train_torch_binary_classifier("transformer", model, features, target)
