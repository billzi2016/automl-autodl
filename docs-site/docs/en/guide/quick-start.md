# Quick start

## Install dependencies

Run this from the repository root:

```bash
pip install -r requirements.txt
```

The training environment includes `pandas`, `numpy`, `scikit-learn`, `torch`, `pytorch-tabnet`, `xgboost`, `lightgbm`, and `catboost`. If an optional library is missing, the matching model returns a structured `skipped` result instead of stopping every other model.

The documentation site has its own dependencies in `docs-site/requirements.txt`.

## Run AutoML

```bash
python main.py
```

This command:

1. Loads and preprocesses `data/application_train.csv`.
2. Runs the models listed in `config.AUTOML_MODELS`.
3. Saves the summary to `outputs/automl_grid_search_results.json`.

## Run AutoDL

```bash
python autodl/train.py
```

This command runs MLP, BLSTM, CNN1D, TabNet, and Transformer, then writes:

```text
outputs/autodl_training_results.json
```

## Docker

Build:

```bash
docker build -t automl-autodl .
```

Run AutoML:

```bash
docker run --rm automl-autodl
```

Run AutoDL:

```bash
docker run --rm automl-autodl python autodl/train.py
```

The Docker path mainly covers CUDA and CPU. MPS detection remains in the code for native Apple Silicon PyTorch runs.

## Documentation site

```bash
cd docs-site
pip install -r requirements.txt
mkdocs serve
```

Strict build:

```bash
mkdocs build --strict
```
