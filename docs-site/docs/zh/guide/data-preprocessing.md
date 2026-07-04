# 数据与预处理

预处理入口在 `preprocess.py`。AutoML 和 AutoDL 都调用同一个入口：

```python
features, target = prepare_training_data()
```

这样做的好处很简单：模型不同，但特征定义不漂移。

## 数据入口

默认数据文件：

```text
data/application_train.csv
```

标签列：

```text
TARGET
```

路径和列名都在 `config.py` 里维护：

- `TRAIN_DATA_PATH`
- `TARGET_COLUMN`
- `ONEHOT_COLUMNS`
- `SCALE_COLUMNS`

## 配置校验

`validate_feature_config()` 会先检查配置里的列是否真实存在。缺列时直接抛出 `ValueError`，错误里会列出缺失字段。

这个校验放在训练前面，避免模型跑到一半才发现字段名写错。

## 类别特征

类别列来自 `config.ONEHOT_COLUMNS`，包括合同类型、性别、是否有车、收入类型、教育程度、职业、组织类型等字段。

处理方式：

```python
features[config.ONEHOT_COLUMNS]
    .fillna(config.MISSING_CATEGORY_VALUE)
    .astype(str)
```

缺失类别统一填成 `Unknown`，再用 `pd.get_dummies()` 做 one-hot。这里没有给缺失值单独开 `dummy_na=True`，因为缺失已经被显式转成了一个类别。

## 数值特征

数值列来自 `config.SCALE_COLUMNS`，包括收入、贷款金额、年金、出生天数、外部评分、社交圈统计和征信查询次数等字段。

处理方式：

- 用 `pd.to_numeric(errors="coerce")` 转成数值。
- 缺失值用列中位数填充。
- 使用 `StandardScaler` 标准化。

标准化对 Logistic Regression、SVM、KNN 这类模型尤其有影响。树模型不依赖特征尺度，但统一预处理可以让所有模型消费同一份矩阵。

## 透传列

不在 `ONEHOT_COLUMNS`、也不在 `SCALE_COLUMNS` 的特征会进入透传逻辑：

- 复制出来。
- 尝试转成数值。
- 缺失值用中位数填充。

这部分设计比较务实：配置里明确列出的字段走明确处理，剩下的字段只要能转成数值，就保留给模型。

## 最终输出

最终特征矩阵由三部分拼接：

- 标准化后的数值列。
- 数值化后的透传列。
- one-hot 后的类别列。

返回结果：

```python
return processed_features, target
```

`processed_features` 会被转成 `float`，方便 scikit-learn、PyTorch 和 TabNet 统一消费。
