# AutoML-AutoDL

`AutoML-AutoDL` 是一个表格风控建模项目，数据入口是 `data/application_train.csv`，标签列是 `TARGET`。项目把同一份预处理结果交给两套模型：一套是传统机器学习模型，统一用 `GridSearchCV` 搜索参数；另一套是基于 PyTorch 和 TabNet 的深度学习表格模型。

最短运行路径：

```bash
pip install -r requirements.txt
python main.py
python autodl/train.py
```

AutoML 入口会运行 `config.AUTOML_MODELS` 里的 13 个模型，并把结果写到：

```text
outputs/automl_grid_search_results.json
```

AutoDL 入口会运行 `config.AUTODL_MODELS` 里的 5 个模型，并把结果写到：

```text
outputs/autodl_training_results.json
```

## 站点内容

- 快速开始：安装依赖、运行 AutoML、运行 AutoDL、使用 Docker。
- 数据与预处理：类别列、数值列、缺失值、one-hot、标准化和透传列。
- AutoML：统一搜索逻辑、`GridSearchCV` 配置、13 个模型的参数网格。
- AutoDL：统一训练器、验证集划分、早停、5 个深度模型结构。
- 项目：设备选择、输出 JSON、目录职责和 PRD。

## 当前模型覆盖

AutoML 模型包括 Logistic Regression、SVM、KNN、GaussianNB、Decision Tree、Random Forest、Extra Trees、Gradient Boosting、HistGradientBoosting、AdaBoost、XGBoost、LightGBM 和 CatBoost。

AutoDL 模型包括 MLP、BLSTM、CNN1D、TabNet 和 Transformer。
