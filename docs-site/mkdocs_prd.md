# AutoML-AutoDL MkDocs 文档站点 PRD

## 1. 文档目的

本 PRD 用来指导 `AutoML-AutoDL` 项目的文档站点建设。站点不是通用模板，也不是只放 README 的壳子。它要把当前仓库里已经实现的表格风控建模流程讲清楚：数据怎么处理，AutoML 怎么搜索，AutoDL 怎么训练，结果怎么保存，Docker 和设备选择怎么配合。

文档写作要贴近代码事实。没有实现的能力不要写成已经支持；已经写进项目的模型、参数和输出路径，要在文档里讲到。

## 2. 项目背景

`AutoML-AutoDL` 是一个面向 Home Credit Default Risk 表格二分类任务的建模工作区。项目包含三条主线：

- 共享预处理流程：读取 `data/application_train.csv`，处理 `TARGET`、类别列、数值列和剩余透传列。
- AutoML：13 个传统机器学习模型，每个模型单独一个文件，统一通过 `GridSearchCV` 搜索参数。
- AutoDL：5 个深度学习表格模型，复用同一份特征矩阵和训练工具。

站点要服务两类读者：

- 想快速运行项目的人，需要清楚的安装、运行、Docker 和输出说明。
- 想看算法实现的人，需要知道每个模型在这个项目里怎么接入、调哪些参数、用什么指标判断效果。

## 3. 建设目标

文档站点放在 `docs-site/` 下，使用 MkDocs 构建，支持中文和英文两套内容。

本次交付要达到这些目标：

- 建立可直接构建的 MkDocs 文档工程。
- 使用 `mkdocs-static-i18n` 支持中英双语。
- 中文为默认语言，英文为平行内容。
- 文档结构和项目结构一致，不写泛泛的 AutoML 概念页。
- 对 `GridSearchCV`、AutoML 模型、AutoDL 模型、预处理、设备策略和输出结果做足够细的说明。
- 提供可由 GitHub Actions 部署到 GitHub Pages 的结构。

## 4. 技术选型

### 4.1 文档框架

- 使用 `MkDocs`。
- 使用 `Material for MkDocs` 作为主题。
- 使用 `mkdocs-static-i18n` 做语言构建和语言切换。
- 文档依赖写入 `docs-site/requirements.txt`。

### 4.2 目录约定

```text
docs-site/
├── mkdocs.yml
├── requirements.txt
├── mkdocs_prd.md
├── github_action_prd.md
└── docs/
    ├── zh/
    │   ├── index.md
    │   ├── guide/
    │   ├── algorithms/
    │   └── project/
    ├── en/
    │   ├── index.md
    │   ├── guide/
    │   ├── algorithms/
    │   └── project/
    └── assets/
```

文档工程要尽量收敛在 `docs-site/`。GitHub Actions 工作流因为 GitHub 的约定，需要放在仓库根目录的 `.github/workflows/`。

## 5. 信息架构

站点需要包含这些页面：

- 首页：项目是什么、当前覆盖哪些模型、最短运行路径。
- 快速开始：依赖安装、AutoML 运行、AutoDL 运行、Docker 运行。
- 数据与预处理：`TARGET`、`ONEHOT_COLUMNS`、`SCALE_COLUMNS`、缺失值、one-hot、标准化、透传列。
- AutoML 总览：统一入口、模型调度、`GridSearchCV` 公共配置、结果结构。
- AutoML 算法详解：13 个模型逐个说明。
- AutoDL 总览：统一训练器、验证集划分、loss、optimizer、early stopping、设备选择。
- AutoDL 模型详解：MLP、BLSTM、CNN1D、TabNet、Transformer。
- 设备与输出：CUDA、MPS、CPU，XGBoost/LightGBM/CatBoost 的运行参数，输出 JSON。
- 项目结构：仓库目录、关键文件职责。
- PRD：保留站点 PRD 和 GitHub Actions PRD 的说明入口。

## 6. 内容要求

### 6.1 写作风格

文档要直接、具体、少套话。避免这些写法：

- “打造完整生态”“赋能建模流程”“具有重要意义”等空泛表达。
- “不仅是 A，更是 B”这类硬拗句式。
- 每段都用三点式总结。
- 用“关键、重要、显著、深度、全面”堆语气，但不给代码细节。

可以保留必要的技术解释，但要写得像项目维护者在说明代码，而不是像宣传页。

### 6.2 代码一致性

文档必须和当前代码保持一致：

- 数据路径：`data/application_train.csv`
- 标签列：`TARGET`
- AutoML 入口：`python main.py`
- AutoDL 入口：`python autodl/train.py`
- AutoML 输出：`outputs/automl_grid_search_results.json`
- AutoDL 输出：`outputs/autodl_training_results.json`
- AutoML 模型列表来自 `config.AUTOML_MODELS`
- AutoDL 模型列表来自 `config.AUTODL_MODELS`
- 统一指标来自 `config.SCORING`
- 最终选参指标来自 `config.REFIT_METRIC`

### 6.3 算法详解深度

AutoML 算法页不能只列模型名。每个模型至少说明：

- 模型在这个项目里的角色。
- 当前 estimator 的关键初始化参数。
- 当前 `param_grid` 搜索范围。
- 和表格风控数据结合时应该观察什么。
- 和统一 `GridSearchCV` 配置的关系。

AutoDL 模型页至少说明：

- 输入张量如何从表格特征得到。
- 模型结构如何处理表格输入。
- 训练配置使用哪些全局参数。
- 输出结果如何进入统一 JSON 结构。
- 当前实现的限制。

## 7. GridSearchCV 说明要求

`GridSearchCV` 是站点必须讲清楚的部分。文档要说明：

- `run_grid_search()` 位于 `utils/train_utils.py`。
- 交叉验证使用 `StratifiedKFold`。
- 折数来自 `config.CV_FOLDS`，当前是 3。
- `shuffle=True`，`random_state=config.RANDOM_STATE`。
- `scoring` 同时计算 `f1`、`roc_auc`、`accuracy`。
- `refit=config.REFIT_METRIC`，当前用 `roc_auc` 选最佳参数。
- `n_jobs=config.GRIDSEARCH_N_JOBS`，当前为 `-1`。
- 树模型和 boosting 库模型内部常设为单线程，是为了把并行资源留给外层搜索。

## 8. 双语要求

- 中文和英文页面结构保持一致。
- 中文可以更细，英文也必须覆盖同样事实，不允许英文只写摘要。
- 中文为默认语言。
- 两种语言都要能从首页进入主要页面。

## 9. 自动化要求

站点要配套 GitHub Actions：

- 只在文档相关文件变化时自动触发。
- 支持 `workflow_dispatch` 手动触发。
- 从 `docs-site/requirements.txt` 安装构建依赖。
- 在 `docs-site/` 内执行 `mkdocs build --strict`。
- 使用 GitHub Pages 官方 Action 上传并部署构建产物。

自动化细节以 `github_action_prd.md` 为准。

## 10. 交付物

本次应交付：

- `docs-site/mkdocs.yml`
- `docs-site/requirements.txt`
- `docs-site/docs/zh/**`
- `docs-site/docs/en/**`
- `docs-site/mkdocs_prd.md`
- `docs-site/github_action_prd.md`
- `.github/workflows/docs.yml`

## 11. 验收标准

满足以下条件即可认为文档站点第一版完成：

1. `docs-site/` 是一个可构建的 MkDocs 工程。
2. 中文和英文页面都存在，并且导航结构一致。
3. 文档准确描述当前项目的预处理、AutoML、AutoDL、设备和输出。
4. AutoML 算法页覆盖 13 个模型，并写明参数网格。
5. AutoDL 算法页覆盖 5 个模型，并写明结构和训练方式。
6. `GridSearchCV` 页面讲清楚交叉验证、指标、refit 和并行策略。
7. GitHub Actions 工作流能按文档变更触发并部署 GitHub Pages。
8. 文档没有明显宣传腔或空泛套话。
