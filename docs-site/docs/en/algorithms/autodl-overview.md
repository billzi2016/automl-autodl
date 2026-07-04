# AutoDL overview

The AutoDL entrypoint is `autodl/train.py`. It reuses `prepare_training_data()` and then runs the models listed in `config.AUTODL_MODELS`:

```text
mlp
blstm
cnn1d
tabnet
transformer
```

## Shared PyTorch trainer

MLP, BLSTM, CNN1D, and Transformer use `train_torch_binary_classifier()`.

The trainer:

- converts the feature matrix to `float32`
- creates a stratified train/validation split
- builds `DataLoader` objects
- trains with `BCEWithLogitsLoss`
- updates parameters with `AdamW`
- evaluates F1, ROC AUC, and Accuracy after each epoch
- keeps the best validation AUC state
- stops early when AUC stops improving

Main settings:

```python
AUTODL_BATCH_SIZE = 512
AUTODL_EPOCHS = 12
AUTODL_LEARNING_RATE = 1e-3
AUTODL_WEIGHT_DECAY = 1e-4
AUTODL_HIDDEN_DIM = 128
AUTODL_DROPOUT = 0.2
AUTODL_EARLY_STOPPING_PATIENCE = 3
```

TabNet uses `pytorch-tabnet` and its own `fit()` method, while still using the project’s split and metric utilities.
