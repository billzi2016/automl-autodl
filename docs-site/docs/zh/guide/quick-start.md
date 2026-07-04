# 快速开始

## 安装依赖

在仓库根目录安装训练依赖：

```bash
pip install -r requirements.txt
```

依赖里包含 `pandas`、`numpy`、`scikit-learn`、`torch`、`pytorch-tabnet`、`xgboost`、`lightgbm` 和 `catboost`。如果某些可选库没装，对应模型会返回 `skipped` 结果，而不是让整个流程直接崩掉。

文档站点自己的依赖在 `docs-site/requirements.txt`。它只用于构建文档，不参与模型训练。

## 运行 AutoML

```bash
python main.py
```

这条命令会做三件事：

1. 调用 `prepare_training_data()` 读取并预处理 `data/application_train.csv`。
2. 按 `config.AUTOML_MODELS` 的顺序运行传统机器学习模型。
3. 打印摘要，并把结果保存到 `outputs/automl_grid_search_results.json`。

## 运行 AutoDL

```bash
python autodl/train.py
```

AutoDL 和 AutoML 复用同一份预处理逻辑。入口会运行 MLP、BLSTM、CNN1D、TabNet 和 Transformer，并把结果保存到：

```text
outputs/autodl_training_results.json
```

## Docker

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

Docker 环境主要按 `CUDA / CPU` 处理。Apple Silicon 的 `MPS` 检测保留在代码里，但更适合在 macOS 宿主机原生跑 PyTorch，而不是放进标准 Linux 容器里期待它生效。

## 构建文档站点

```bash
cd docs-site
pip install -r requirements.txt
mkdocs serve
```

严格构建：

```bash
mkdocs build --strict
```
