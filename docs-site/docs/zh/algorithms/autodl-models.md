# AutoDL 模型详解

AutoDL 模型都接收同一份 dense tabular feature matrix。也就是说，模型看到的不是原始字符串类别，而是预处理后的浮点矩阵：

```text
[batch_size, feature_dim]
```

训练目标都是二分类：根据一行申请人的特征预测 `TARGET`。PyTorch 模型输出一个 logit，再通过 sigmoid 变成正类概率。

## 深度学习训练在这里怎么跑

MLP、BLSTM、CNN1D 和 Transformer 共用 `train_torch_binary_classifier()`。

训练过程可以按这条线理解：

1. 把 pandas DataFrame 转成 `float32` numpy array。
2. 用 `train_test_split(..., stratify=target)` 切训练集和验证集。
3. 用 `DataLoader` 分 batch。
4. 模型前向计算 logits。
5. 用 `BCEWithLogitsLoss` 计算二分类损失。
6. 用 `AdamW` 更新参数。
7. 每轮结束后在验证集算 F1、AUC、Accuracy。
8. 验证 AUC 不再提升时触发 early stopping。

关键配置：

```python
AUTODL_BATCH_SIZE = 512
AUTODL_EPOCHS = 12
AUTODL_LEARNING_RATE = 1e-3
AUTODL_WEIGHT_DECAY = 1e-4
AUTODL_HIDDEN_DIM = 128
AUTODL_DROPOUT = 0.2
AUTODL_EARLY_STOPPING_PATIENCE = 3
```

## MLP

文件：`autodl/mlp.py`

MLP，全称 multilayer perceptron，也就是多层全连接神经网络。它把每一行表格特征当作一个向量，然后一层一层做线性变换和非线性变换。

结构：

```text
Linear(input_dim -> 128)
ReLU
Dropout(0.2)
Linear(128 -> 64)
ReLU
Dropout(0.2)
Linear(64 -> 1)
```

它怎么学习：

- 第一层把原始特征组合成 128 个隐藏表示。
- ReLU 把线性模型变成非线性模型。
- Dropout 在训练时随机丢掉一部分隐藏单元，降低过拟合。
- 最后一层输出一个 logit。
- loss 根据 logit 和真实 `TARGET` 调整所有层的权重。

MLP 不关心特征顺序，也不显式建树。它会学“特征加权组合之后再组合”的模式。比如金额、外部评分、年龄、职业 one-hot 之间可能形成某种非线性组合，MLP 可以尝试捕捉。

在这个项目里，MLP 是 AutoDL 的基础模型。如果 MLP 明显弱于树模型，不奇怪；很多表格任务里，GBDT 仍然很强。

## BLSTM

文件：`autodl/blstm.py`

LSTM 原本常用于序列，比如文本或时间序列。这里的 BLSTM 把一行表格特征临时 reshape 成一个“特征序列”：

```python
sequence = features.unsqueeze(-1)
```

形状变化：

```text
[batch_size, feature_dim]
-> [batch_size, feature_dim, 1]
```

模型结构：

```text
2-layer bidirectional LSTM
take last timestep output
Linear(128 -> 64)
ReLU
Dropout(0.2)
Linear(64 -> 1)
```

它怎么学习：

- LSTM 会从左到右读特征。
- 双向 LSTM 还会从右到左读一遍。
- 每个位置的隐藏状态会带着前后特征的信息。
- 最后取最后一个 timestep 的输出做分类。

这里要注意一个边界：表格特征列不是天然时间序列。`EXT_SOURCE_1`、`AMT_CREDIT`、`OCCUPATION_TYPE_xxx` 之间没有像句子单词那样的自然顺序。所以 BLSTM 在这里更像一个实验 baseline：它能学习特征间的顺序化交互，但结果要谨慎解释。

## CNN1D

文件：`autodl/cnn1d.py`

CNN1D 把一行表格特征当作一条一维信号：

```python
sequence = features.unsqueeze(1)
```

形状变化：

```text
[batch_size, feature_dim]
-> [batch_size, 1, feature_dim]
```

结构：

```text
Conv1d(1 -> 32, kernel_size=3, padding=1)
ReLU
Conv1d(32 -> 64, kernel_size=3, padding=1)
ReLU
AdaptiveAvgPool1d(1)
Flatten
Linear(64 -> 64)
ReLU
Dropout(0.2)
Linear(64 -> 1)
```

它怎么学习：

- 卷积核每次看相邻的几个特征。
- 第一层卷积学简单局部组合。
- 第二层卷积在第一层基础上学更抽象的局部组合。
- `AdaptiveAvgPool1d(1)` 把整条特征序列压成固定长度表示。
- 最后的全连接层做二分类。

CNN1D 的前提是“相邻特征之间可能有局部模式”。在图像里这个前提很自然，像素旁边还是像素；在表格里不一定成立，因为列顺序往往是人为排列的。所以 CNN1D 适合作为结构实验，不能简单理解成它一定能发现真正的局部空间规律。

## TabNet

文件：`autodl/tabnet.py`

TabNet 是专门面向表格数据设计的深度模型。它的一个核心想法是：模型在每一步只关注一部分特征，而不是每次都把所有特征一股脑丢进去。

这个项目没有手写 TabNet，而是调用 `pytorch_tabnet.tab_model.TabNetClassifier`。

训练方式：

```python
TabNetClassifier(
    seed=config.RANDOM_STATE,
    verbose=0,
    device_name=device_name,
    optimizer_params={"lr": config.AUTODL_LEARNING_RATE},
)
```

fit 配置：

```python
max_epochs=config.AUTODL_EPOCHS
batch_size=config.AUTODL_BATCH_SIZE
virtual_batch_size=min(128, config.AUTODL_BATCH_SIZE)
eval_metric=["auc"]
```

它怎么学习：

- TabNet 会通过注意力机制选择特征。
- 多个 decision step 会逐步使用不同特征组合。
- 每一步的输出汇总后用于分类。
- 训练时用验证集 AUC 监控效果。

当前实现里，TabNet 接收的是预处理后的 dense matrix。类别特征已经 one-hot，因此它不是直接处理原始类别字段。

设备逻辑：

```python
device_name = "cuda" if str(device) == "cuda" else "cpu"
```

也就是说，TabNet 这里只显式使用 CUDA 或 CPU，不走 MPS。

## Transformer

文件：`autodl/transformer.py`

Transformer 原本常用于文本。这里把每个表格特征看成一个 token。

输入处理：

```python
sequence = features.unsqueeze(-1)
embedded = self.embedding(sequence)
```

形状变化可以这样理解：

```text
[batch_size, feature_dim]
-> [batch_size, feature_dim, 1]
-> [batch_size, feature_dim, 64]
```

结构：

```text
Linear(1 -> 64)
TransformerEncoderLayer(
  d_model=64,
  nhead=4,
  dim_feedforward=128,
  dropout=0.2
)
TransformerEncoder(num_layers=2)
mean pooling over feature dimension
Linear(64 -> 64)
ReLU
Dropout(0.2)
Linear(64 -> 1)
```

它怎么学习：

- 每个特征先被投影成 64 维 embedding。
- self-attention 会让每个特征“看”其他特征。
- 多头注意力允许模型从不同角度建模特征关系。
- encoder 输出后，对所有特征位置做平均池化。
- 最后的 MLP head 输出一个 logit。

当前实现没有特征名 embedding，也没有显式位置编码。也就是说，模型知道第几个位置的数值，但没有直接知道“这个位置叫 AMT_CREDIT”。如果后续要做更强的表格 Transformer，可以考虑加入特征名 embedding、类别 embedding 或更适合表格的 tokenizer。

## 怎么比较 AutoDL 模型

这几个模型不是按“越复杂越好”排列的。

- MLP 是 dense 表格深度学习基线。
- BLSTM 测试“把特征当序列”是否有帮助。
- CNN1D 测试“相邻特征局部模式”是否有帮助。
- TabNet 是更贴表格数据设计的深度模型。
- Transformer 测试 self-attention 建模特征交互的效果。

表格数据上，树模型经常很强。AutoDL 的价值不只是追求分数，也在于提供一套和传统模型对照的深度学习实验框架。
