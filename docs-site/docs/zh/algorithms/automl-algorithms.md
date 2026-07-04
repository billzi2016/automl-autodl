# AutoML 算法详解

这一页按当前代码写，也尽量把模型本身讲明白。你可以把它当成“这 13 个传统机器学习模型在本项目里到底怎么工作”的说明。

所有 AutoML 模型都吃同一份输入：

```text
预处理后的 dense feature matrix X
二分类标签 y = TARGET
```

所有模型都走同一个搜索器：

```text
GridSearchCV + StratifiedKFold(3 折) + AUC/F1/Accuracy
```

区别在于：每个模型“学习规律”的方式不同，参数网格也不同。

## Logistic Regression

文件：`automl/logistic_regression.py`

Logistic Regression 是线性二分类模型。它做的事情可以理解成：给每个特征一个权重，把一行样本的特征加权求和，再通过 sigmoid 转成“违约概率”。

如果某个特征的权重大于 0，它会把预测往 `TARGET=1` 推；如果权重小于 0，它会把预测往 `TARGET=0` 推。这个模型训练时会不断调整这些权重，让预测概率和真实标签之间的误差变小。

初始化：

```python
LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    random_state=config.RANDOM_STATE,
)
```

参数网格：

```python
{
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
}
```

参数怎么理解：

- `C` 是正则强度的反向表达。`C` 越小，模型越保守，权重越不容易变大；`C` 越大，模型越愿意贴合训练数据。
- `penalty="l1"` 会把一部分特征权重压到 0，效果有点像自动筛特征。
- `penalty="l2"` 会让权重整体变小，但通常不会直接变成 0。
- `class_weight="balanced"` 会给少数类更高权重，适合违约样本比例较低的情况。

在这个项目里，Logistic Regression 是非常重要的基线。它不一定分数最高，但它能告诉你：经过 one-hot 和标准化后，线性关系能做到什么程度。

## SVM

文件：`automl/svm.py`

SVM 的核心想法是找一条分界线，或者在高维空间里找一个分界面，把两类样本尽量分开。它不只是要求分对，还希望分界面离两边样本都远一点。这个“远一点”的间隔叫 margin。

线性核 SVM 就是在原始特征空间里找分界面。RBF 核会把样本映射到更复杂的空间，让模型能学习弯曲边界。

初始化：

```python
SVC(random_state=config.RANDOM_STATE)
```

参数网格：

```python
{
    "C": [0.5, 1.0, 2.0],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale"],
    "class_weight": [None, "balanced"],
}
```

参数怎么理解：

- `C` 越大，模型越不愿意容忍训练错误，边界可能更贴训练集。
- `kernel="linear"` 偏简单，速度和解释性更好。
- `kernel="rbf"` 更灵活，但训练更慢，也更容易受参数和数据规模影响。
- `gamma="scale"` 是 scikit-learn 的常用默认策略，按特征数量和方差自动定核宽度。
- `class_weight="balanced"` 用于类别不均衡。

SVM 对特征尺度很敏感。项目里数值列先做 `StandardScaler`，这一步对 SVM 很关键，否则金额类字段可能会压过其他字段。

## KNN

文件：`automl/knn.py`

KNN 几乎不“训练”参数。它的做法是把训练样本记下来。预测新样本时，先找离它最近的 K 个训练样本，再看这些邻居大多属于哪一类。

如果一个申请人的特征向量和若干已知违约样本很近，KNN 就更倾向于预测违约。这里的“近”由距离函数定义。

初始化：

```python
KNeighborsClassifier(n_jobs=1)
```

参数网格：

```python
{
    "n_neighbors": [5, 11, 21],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
```

参数怎么理解：

- `n_neighbors=5` 更看局部，容易受噪声影响。
- `n_neighbors=21` 更平滑，但可能把局部差异抹掉。
- `weights="uniform"` 表示邻居一票一票算。
- `weights="distance"` 表示越近的邻居权重越大。
- `p=1` 是曼哈顿距离，按各维差值绝对值累加。
- `p=2` 是欧氏距离，也就是常见的直线距离。

KNN 很依赖标准化，也怕高维稀疏特征。one-hot 之后特征维度会上升，所以它在这里更适合作为距离模型的对照，不一定适合作最终模型。

## Gaussian Naive Bayes

文件：`automl/naive_bayes.py`

朴素贝叶斯用概率来分类。它会估计：在违约样本里，每个特征大概长什么样；在非违约样本里，每个特征又大概长什么样。预测时，它比较“这行样本更像违约类，还是更像非违约类”。

“朴素”的意思是它假设特征之间条件独立。现实里这个假设通常不完全成立，比如收入、贷款金额、教育程度之间明显有关。但这个模型速度快，能提供一个轻量 baseline。

初始化：

```python
GaussianNB()
```

参数网格：

```python
{
    "var_smoothing": [1e-9, 1e-8, 1e-7],
}
```

`var_smoothing` 会给方差加一个很小的稳定项，避免某些特征方差太小导致数值问题。它不负责让模型变复杂，只是让概率计算更稳。

## Decision Tree

文件：`automl/decision_tree.py`

决策树像一串 if-else。训练时，模型会不断选择一个特征和一个切分点，把样本分成两边，让分出来的两组标签更“纯”。

举个简化例子：模型可能先按 `EXT_SOURCE_2` 切一刀，再在某个分支里按 `DAYS_BIRTH` 切一刀，再继续按贷款金额或职业 one-hot 特征切。最后每个叶子节点里会落入一批样本，叶子里正负样本比例就变成预测概率。

初始化：

```python
DecisionTreeClassifier(random_state=config.RANDOM_STATE)
```

参数网格：

```python
{
    "max_depth": [4, 8, 12, None],
    "min_samples_split": [2, 10, 30],
    "min_samples_leaf": [1, 5, 10],
    "class_weight": [None, "balanced"],
}
```

参数怎么理解：

- `max_depth` 控制树最多长多深。深度越大，模型越容易记住训练集细节。
- `min_samples_split` 控制一个节点至少有多少样本才允许继续切。
- `min_samples_leaf` 控制叶子节点至少保留多少样本，能防止叶子只记住一两个样本。
- `class_weight="balanced"` 会提高少数类在切分时的影响。

单棵树的优势是好解释，缺点是容易过拟合。它在这里适合帮助理解树模型的基本行为。

## Random Forest

文件：`automl/random_forest.py`

Random Forest 是很多棵决策树的集合。每棵树会看到一部分随机抽样的训练数据，也会在切分时只看一部分随机特征。最后分类时，多棵树一起投票或平均概率。

它解决的是单棵树太不稳定的问题。单棵树可能因为某些样本扰动就长得完全不同，多棵树平均后会稳很多。

初始化：

```python
RandomForestClassifier(
    random_state=config.RANDOM_STATE,
    n_jobs=1,
)
```

参数网格：

```python
{
    "n_estimators": [200, 400],
    "max_depth": [8, 16, None],
    "min_samples_split": [2, 10],
    "class_weight": [None, "balanced_subsample"],
}
```

参数怎么理解：

- `n_estimators` 是树的数量。更多树通常更稳，但训练更慢。
- `max_depth` 控制每棵树的复杂度。
- `min_samples_split` 控制树继续分裂的门槛。
- `balanced_subsample` 会在每棵树的 bootstrap 样本上计算类别权重。

项目里把模型内部 `n_jobs=1`，因为外层 `GridSearchCV` 已经用 `n_jobs=-1` 并行跑参数组合。

## Extra Trees

文件：`automl/extra_trees.py`

Extra Trees 也是很多棵树的集合，但它比 Random Forest 更随机。随机森林会在候选特征里寻找较好的切分点，Extra Trees 会更随机地选切分点。

这种更强的随机性会降低方差，也可能增加偏差。简单说，它更不容易死记训练集，但也可能切得没那么精细。

初始化：

```python
ExtraTreesClassifier(
    random_state=config.RANDOM_STATE,
    n_jobs=1,
)
```

参数网格：

```python
{
    "n_estimators": [200, 400],
    "max_depth": [8, 16, None],
    "min_samples_split": [2, 10],
    "class_weight": [None, "balanced"],
}
```

看结果时，可以把 Extra Trees 和 Random Forest 放在一起。如果 Extra Trees 分数接近甚至更好，说明这个数据上更强随机化没有伤害模型；如果明显更差，说明随机切分丢掉了不少有用结构。

## Gradient Boosting

文件：`automl/gradient_boosting.py`

Gradient Boosting 也是树模型集合，但它不是“一堆树同时投票”，而是一棵接一棵地补错。

第一棵树先给一个粗略预测。第二棵树去学习第一棵树没处理好的部分。后面的树继续补前面模型的残差。最终很多棵弱树加起来，形成一个强模型。

初始化：

```python
GradientBoostingClassifier(random_state=config.RANDOM_STATE)
```

参数网格：

```python
{
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0],
}
```

参数怎么理解：

- `n_estimators` 是 boosting 轮数，也就是树的数量。
- `learning_rate` 控制每棵树修正前面错误的步子多大。
- `max_depth` 控制每棵弱树的复杂度。boosting 常用浅树。
- `subsample=0.8` 表示每轮只用 80% 样本，能增加随机性。

`learning_rate` 和 `n_estimators` 要一起看。小学习率通常需要更多树，大学习率更快但更容易过拟合。

## HistGradientBoosting

文件：`automl/hist_gradient_boosting.py`

HistGradientBoosting 也是梯度提升树，但它用直方图方式处理连续特征。可以理解成先把连续值分桶，再在桶上找切分点。这样训练通常更快，也更适合较大的表格数据。

初始化：

```python
HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)
```

参数网格：

```python
{
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 8, None],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],
}
```

参数怎么理解：

- `max_leaf_nodes` 限制叶子节点数量，比单纯限制深度更直接地控制树复杂度。
- `min_samples_leaf` 要求每个叶子有足够样本，避免切出很小的叶子。
- `max_depth=None` 不代表完全无限制，因为还有叶子数和叶子样本数约束。

它适合作为 scikit-learn 体系里更现代的 GBDT baseline。

## AdaBoost

文件：`automl/adaboost.py`

AdaBoost 也是一轮一轮训练。它会提高前一轮分错样本的权重，让下一轮模型更关注这些难样本。

可以把它想成：模型先做一版，错的样本被圈出来加权；下一版模型必须更努力处理这些错样本；多轮以后，把每一轮模型按表现加权组合起来。

初始化：

```python
AdaBoostClassifier(random_state=config.RANDOM_STATE)
```

参数网格：

```python
{
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 1.0],
}
```

参数怎么理解：

- `n_estimators` 是弱学习器数量。
- `learning_rate` 控制每一轮模型的贡献幅度。

AdaBoost 对噪声比较敏感。如果某些标签本身有问题，它可能会反复追着这些难样本学，导致泛化变差。

## XGBoost

文件：`automl/xgboost_model.py`

XGBoost 是工程化很强的梯度提升树。它也是一棵棵树补前面模型的错误，但加入了更完整的正则、采样、并行和缺失处理机制。

在这个项目里，XGBoost 接收的是预处理后的 dense 特征矩阵。缺少 `xgboost` 时，模型会返回 `skipped`，不会中断其他模型。

初始化重点：

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=config.RANDOM_STATE,
    n_jobs=1,
    **runtime_params,
)
```

参数网格：

```python
{
    "n_estimators": [200, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
```

参数怎么理解：

- `n_estimators` 是树的数量。
- `max_depth` 控制每棵树多深。
- `learning_rate` 控制每棵树的贡献。
- `subsample` 每轮抽样一部分样本。
- `colsample_bytree` 每棵树抽样一部分特征。

有 NVIDIA 环境时，运行参数会设置：

```python
{
    "tree_method": "hist",
    "device": "cuda",
}
```

否则走 CPU 的 `hist`。

## LightGBM

文件：`automl/lightgbm_model.py`

LightGBM 也是梯度提升树，特点是训练速度快，常用于大规模表格数据。它常被拿来和 XGBoost 对比。

LightGBM 的树生长方式更偏 leaf-wise：它倾向于优先分裂能带来最大收益的叶子。这通常很强，但如果不限制复杂度，也容易过拟合。

初始化重点：

```python
LGBMClassifier(
    objective="binary",
    random_state=config.RANDOM_STATE,
    n_jobs=1,
    verbose=-1,
    **runtime_params,
)
```

参数网格：

```python
{
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.1],
    "max_depth": [-1, 8, 12],
    "num_leaves": [31, 63],
    "subsample": [0.8, 1.0],
}
```

参数怎么理解：

- `num_leaves` 是最关键的复杂度参数之一。叶子越多，模型能表达的规则越复杂。
- `max_depth=-1` 表示不按深度硬限制，但仍会受 `num_leaves` 限制。
- `subsample` 用来降低过拟合风险。

检测到 CUDA 分支时会设置：

```python
{
    "device_type": "gpu",
    "gpu_device_id": 0,
}
```

其他情况走 CPU。

## CatBoost

文件：`automl/catboost_model.py`

CatBoost 也是 boosting 树模型。它原本很擅长直接处理类别特征，但当前项目在预处理阶段已经做了 one-hot，所以这里的 CatBoost 吃的是统一 dense 特征矩阵。

它仍然适合作为强基线，因为 CatBoost 的树构建和正则策略在很多表格任务上表现稳定。

初始化重点：

```python
CatBoostClassifier(
    random_seed=config.RANDOM_STATE,
    verbose=0,
    thread_count=1,
    loss_function="Logloss",
    **runtime_params,
)
```

参数网格：

```python
{
    "iterations": [200, 400],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "l2_leaf_reg": [3, 5, 7],
}
```

参数怎么理解：

- `iterations` 是 boosting 轮数。
- `depth` 控制树深度。
- `learning_rate` 控制每轮更新幅度。
- `l2_leaf_reg` 是叶子值的 L2 正则，值越大越保守。

CUDA 环境下会设置：

```python
{
    "task_type": "GPU",
    "devices": "0",
}
```

否则走 CPU。

## 怎么看这些模型的结果

不要只看单个模型的一次分数。更合理的看法是：

- Logistic Regression 给线性基线。
- KNN 和 Naive Bayes 给轻量对照。
- Decision Tree 给可解释树基线。
- Random Forest 和 Extra Trees 看 bagging 集成效果。
- Gradient Boosting、HistGradientBoosting、AdaBoost 看 sklearn 体系的 boosting 效果。
- XGBoost、LightGBM、CatBoost 看工业常用 GBDT 库的表现。

项目用 AUC 选参。AUC 高说明模型排序能力更好，但最终业务阈值仍然需要单独定。F1 和 Accuracy 是辅助指标，不应该脱离样本比例单独解释。
