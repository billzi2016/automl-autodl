# AutoML-AutoDL 文档站点 GitHub Actions PRD

## 1. 文档目的

本 PRD 定义 `AutoML-AutoDL` 文档站点的 GitHub Actions 构建与部署方案。它只处理 `docs-site/` 下的 MkDocs 文档，不运行模型训练，不构建 Docker 镜像，也不发布 Python 包。

## 2. 部署目标

工作流用于把 `docs-site/` 构建成静态文件，并部署到 GitHub Pages。

目标：

- 文档变更后自动构建。
- 支持手动触发。
- 构建命令可在本地复现。
- 部署过程使用 GitHub Pages 官方 Action。
- 权限只给文档部署需要的范围。

## 3. 触发策略

工作流名建议为 `Deploy documentation`，文件放在：

```text
.github/workflows/docs.yml
```

自动触发范围：

- `docs-site/**`
- `.github/workflows/docs.yml`
- `README.md`
- `README_CN.md`

触发分支：

- `main`
- `master`

同时支持：

- `workflow_dispatch`

不应因为训练代码、数据文件或 Dockerfile 的普通变更就部署文档，除非这些变更同时改了文档相关路径。

## 4. 权限要求

工作流使用 GitHub Pages 官方部署链路，权限设置为：

```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

并发控制：

```yaml
concurrency:
  group: pages
  cancel-in-progress: false
```

这里不取消正在进行的 Pages 部署，避免两个文档版本互相打断后状态不清楚。

## 5. 构建环境

使用 Ubuntu runner 和稳定 Python 版本：

- `ubuntu-latest`
- Python `3.11`

依赖从 `docs-site/requirements.txt` 安装。不要依赖仓库根目录的 `requirements.txt`，因为根目录依赖用于模型训练，里面有 `torch`、`xgboost`、`lightgbm`、`catboost` 等包。文档构建不需要安装这些训练依赖。

## 6. 构建命令

工作目录设为 `docs-site`，执行：

```bash
mkdocs build --strict
```

`--strict` 用来让坏链接、缺页和配置问题尽早失败。文档站点第一版就应该接受这个约束，否则后面页面变多时更难排查。

构建产物路径：

```text
docs-site/site
```

## 7. 部署步骤

工作流应使用：

- `actions/checkout`
- `actions/setup-python`
- `actions/configure-pages`
- `actions/upload-pages-artifact`
- `actions/deploy-pages`

部署 job 需要依赖 build job。build job 负责安装依赖和构建产物，deploy job 负责发布 GitHub Pages。

## 8. 非目标

本工作流不做这些事：

- 不运行 `python main.py`
- 不运行 `python autodl/train.py`
- 不下载或训练 Home Credit 全量数据
- 不构建 Docker 镜像
- 不上传模型结果 JSON
- 不发布 PyPI 包
- 不做代码测试矩阵

如果后续需要训练 CI，应单独建工作流，不要塞进文档部署。

## 9. 验收标准

1. 仓库存在 `.github/workflows/docs.yml`。
2. 工作流只面向文档站点构建部署。
3. 文档相关路径变更会自动触发。
4. 支持手动触发。
5. 构建依赖来自 `docs-site/requirements.txt`。
6. 构建命令为 `mkdocs build --strict`。
7. 部署目标为 GitHub Pages。
8. 工作流权限不超过 Pages 部署需要的范围。
