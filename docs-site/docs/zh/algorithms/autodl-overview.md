# AutoDL 总览

AutoDL 入口是 `autodl/train.py`。它和 AutoML 一样先调用 `prepare_training_data()`，然后按 `config.AUTODL_MODELS` 调度模型。

当前模型列表：

```text
mlp
blstm
cnn1d
tabnet
transformer
```

## 统一训练器

MLP、BLSTM、CNN1D 和 Transformer 都走 `train_torch_binary_classifier()`。

训练逻辑：

- 把 pandas 特征矩阵转成 `float32` numpy array。
- 使用 `train_test_split()` 划分训练集和验证集。
- `test_size=config.AUTODL_VALIDATION_SIZE`，当前是 0.2。
- 使用 `stratify=target_array` 保持标签比例。
- 用 `BCEWithLogitsLoss` 训练二分类模型。
- 优化器是 `AdamW`。
- 每个 epoch 后在验证集上计算 F1、AUC、Accuracy。
- 如果验证 AUC 连续若干轮没有提升，就触发 early stopping。

关键配置来自 `config.py`：

```python
AUTODL_BATCH_SIZE = 512
AUTODL_EPOCHS = 12
AUTODL_LEARNING_RATE = 1e-3
AUTODL_WEIGHT_DECAY = 1e-4
AUTODL_HIDDEN_DIM = 128
AUTODL_DROPOUT = 0.2
AUTODL_EARLY_STOPPING_PATIENCE = 3
```

## TabNet 的区别

TabNet 使用 `pytorch-tabnet` 的 `TabNetClassifier`，没有走通用 PyTorch 训练循环。它仍然复用同样的训练/验证划分和同样的指标计算。

TabNet 配置：

```python
max_epochs=config.AUTODL_EPOCHS
batch_size=config.AUTODL_BATCH_SIZE
virtual_batch_size=min(128, config.AUTODL_BATCH_SIZE)
optimizer_params={"lr": config.AUTODL_LEARNING_RATE}
eval_metric=["auc"]
```

## 输出结构

AutoDL 结果也使用 `build_completed_result()`：

```json
{
  "model_name": "mlp",
  "status": "completed",
  "best_score": 0.0,
  "best_params": {
    "batch_size": 512,
    "epochs": 12,
    "learning_rate": 0.001
  },
  "cv_metrics": {
    "f1": 0.0,
    "roc_auc": 0.0,
    "accuracy": 0.0
  },
  "device": "cpu"
}
```

字段名仍叫 `cv_metrics`，但 AutoDL 这里不是交叉验证结果，而是验证集指标。后续如果要更严谨，可以把 AutoDL 输出字段拆成 `validation_metrics`。
