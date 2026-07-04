# AutoDL models

All AutoDL models receive the same dense matrix:

```text
[batch_size, feature_dim]
```

They predict the binary `TARGET`. PyTorch models output one logit, and sigmoid converts it to the positive-class probability.

## MLP

MLP is a stack of fully connected layers. It treats a row of tabular features as one vector.

Structure:

```text
Linear(input_dim -> 128)
ReLU
Dropout(0.2)
Linear(128 -> 64)
ReLU
Dropout(0.2)
Linear(64 -> 1)
```

It learns weighted feature combinations, then combinations of those combinations. It is the basic deep learning baseline for dense tabular input.

## BLSTM

BLSTM reshapes a feature row into a sequence:

```text
[batch_size, feature_dim]
-> [batch_size, feature_dim, 1]
```

A bidirectional LSTM reads the feature sequence in both directions and uses the final output for classification.

This is experimental for tabular data. Feature columns are not natural time steps, so BLSTM results should be compared carefully with MLP and tree models.

## CNN1D

CNN1D treats a feature row as a one-dimensional signal:

```text
[batch_size, feature_dim]
-> [batch_size, 1, feature_dim]
```

Structure:

```text
Conv1d(1 -> 32)
ReLU
Conv1d(32 -> 64)
ReLU
AdaptiveAvgPool1d(1)
Linear head
```

The convolution layers learn local patterns between neighboring columns. That assumption is natural for signals, but less obvious for tabular data, where column order is designed by the developer.

## TabNet

TabNet is a neural architecture designed for tabular data. It uses attention-like feature selection across multiple decision steps.

The project uses `pytorch_tabnet.tab_model.TabNetClassifier`, not a handwritten TabNet implementation.

Settings include:

```python
max_epochs=config.AUTODL_EPOCHS
batch_size=config.AUTODL_BATCH_SIZE
virtual_batch_size=min(128, config.AUTODL_BATCH_SIZE)
eval_metric=["auc"]
```

The current implementation uses CUDA when available; otherwise it uses CPU.

## Transformer

The Transformer model treats each feature as a token.

Shape:

```text
[batch_size, feature_dim]
-> [batch_size, feature_dim, 1]
-> [batch_size, feature_dim, 64]
```

It applies a Transformer encoder, mean-pools across feature positions, and uses a small classification head.

The current version does not include feature-name embeddings or explicit positional encodings. It is a baseline for self-attention over dense tabular features, not a full specialized tabular Transformer.
