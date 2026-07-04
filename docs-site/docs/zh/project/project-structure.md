# 项目结构

```text
automl-autodl/
├── automl/
├── autodl/
├── data/
├── docs-site/
├── utils/
├── config.py
├── preprocess.py
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
└── README_CN.md
```

## 根目录入口

`main.py` 是 AutoML 入口。它负责预处理、按配置调度模型、汇总结果。

`preprocess.py` 是共享预处理入口。AutoML 和 AutoDL 都调用它。

`config.py` 放项目配置，包括：

- 数据路径和输出路径。
- 标签列。
- one-hot 列。
- 标准化列。
- AutoML 和 AutoDL 模型列表。
- 交叉验证、评分指标和训练参数。

## automl/

每个传统模型一个文件：

```text
adaboost.py
catboost_model.py
decision_tree.py
extra_trees.py
gradient_boosting.py
hist_gradient_boosting.py
knn.py
lightgbm_model.py
logistic_regression.py
naive_bayes.py
random_forest.py
svm.py
xgboost_model.py
```

文件里的主要函数都是 `train_model(features, target)`。

## autodl/

深度模型和统一入口：

```text
blstm.py
cnn1d.py
mlp.py
tabnet.py
transformer.py
train.py
```

`train.py` 是 AutoDL 入口。其他文件负责具体模型结构。

## utils/

`train_utils.py`：

- AutoML 的 `run_grid_search()`。
- AutoDL 的训练、验证、数据集包装和指标计算。

`device_utils.py`：

- CUDA、MPS、CPU 检测。
- XGBoost、LightGBM、CatBoost 的运行参数。

`report_utils.py`：

- 打印结果摘要。
- 保存 JSON 输出。

## docs-site/

文档站点工程。它有自己的 `requirements.txt`，不依赖根目录训练环境。
