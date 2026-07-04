# PRD

文档站点相关 PRD 保留在 `docs-site/` 根目录：

- `mkdocs_prd.md`：定义 AutoML-AutoDL 文档站点要写什么、怎么组织、双语怎么处理。
- `github_action_prd.md`：定义 GitHub Actions 如何构建并部署文档站点。

这两个文件是给维护者和后续执行者看的，不参与模型训练。

## 当前重点

文档站点第一版要先把项目讲准确：

- Home Credit 表格二分类任务。
- 共享预处理流程。
- 13 个 AutoML 模型和参数网格。
- `GridSearchCV` 的交叉验证、指标和 refit 规则。
- 5 个 AutoDL 模型结构。
- CUDA、MPS、CPU 的设备选择边界。
- JSON 输出路径和结果结构。

后续可以再加实验报告、模型分数对比、参数搜索曲线和常见问题。
