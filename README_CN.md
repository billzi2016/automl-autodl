# AutoML-AutoDL 项目说明

语言：[English](README.md) | 简体中文

双语支持：本仓库提供英文和中文 README 及文档说明。

文档站点：https://billzi2016.github.io/automl-autodl/

这是一个面向表格风控任务的统一建模工作区，基于 Home Credit Default Risk 数据集构建。项目包含共享的数据预处理流程、传统机器学习模块，以及深度学习模块，适合用于展示完整的表格建模工程结构。

## 项目结构

```text
automl-autodl/
├── automl/          # 传统机器学习模型与 GridSearchCV 搜索
├── autodl/          # 基于 PyTorch / TabNet 的深度学习模型
├── data/            # 示例数据文件
├── utils/           # 公共训练工具、设备判断、结果输出工具
├── config.py        # 统一配置文件
├── preprocess.py    # 共享预处理流程
├── main.py          # AutoML 入口
├── requirements.txt
├── Dockerfile
└── README.md
```

## 数据处理流程

- 数据文件：`data/application_train.csv`
- 标签列：`TARGET`
- 类别特征列表定义在 `config.ONEHOT_COLUMNS`
- 数值特征列表定义在 `config.SCALE_COLUMNS`

预处理流程包括：

- 缺失值填充
- 类别特征 one-hot 编码
- 数值特征标准化
- 生成统一的稠密特征矩阵，供 AutoML 和 AutoDL 共同使用

## AutoML 模块

`automl/` 目录下每个模型单独一个文件，便于直接展示算法覆盖范围。

当前包含：

- Logistic Regression
- SVM
- KNN
- Naive Bayes
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- AdaBoost
- XGBoost
- LightGBM
- CatBoost

统一特性：

- 使用共享的 `GridSearchCV`
- 3 折交叉验证
- 统一记录 `AUC`、`F1`、`Accuracy`
- 使用统一结果结构输出

运行方式：

```bash
python main.py
```

## AutoDL 模块

`autodl/` 目录下提供多种深度学习表格建模基线：

- `mlp.py`：多层感知机
- `blstm.py`：双向 LSTM
- `cnn1d.py`：一维卷积网络
- `tabnet.py`：TabNet 表格模型
- `transformer.py`：Transformer 编码器结构

统一入口：

```bash
python autodl/train.py
```

## 设备判断逻辑

设备检测集中在 `utils/device_utils.py` 中处理。

逻辑策略如下：

- 优先检测 `CUDA`
- Apple Silicon 环境下保留 `MPS` 检测逻辑
- 不满足 GPU 条件时自动回退到 `CPU`
- `xgboost`、`lightgbm`、`catboost` 只有在兼容分支下才会真正开启 GPU 参数

说明：

- Docker 环境中主要考虑 `CUDA / CPU`
- `MPS` 逻辑仍保留在项目代码里
- Apple Silicon 的 `MPS` 更适合直接在宿主机原生运行，而不是依赖标准 Linux Docker 容器

## Docker 使用方式

构建镜像：

```bash
docker build -t automl-autodl .
```

运行 AutoML：

```bash
docker run --rm automl-autodl
```

运行 AutoDL：

```bash
docker run --rm automl-autodl python autodl/train.py
```

## 环境安装

安装依赖：

```bash
pip install -r requirements.txt
```

## 说明

- `data/application_train.csv` 被保留在仓库中，作为项目展示用示例数据
- `data/` 下其他生成的 CSV 文件默认会被 `.gitignore` 忽略
- 输出结果默认写入 `outputs/`
