# AutoML-AutoDL

`AutoML-AutoDL` is a tabular risk modeling workspace for the Home Credit Default Risk style binary classification task. The input file is `data/application_train.csv`, and the target column is `TARGET`.

The project has one shared preprocessing pipeline and two modeling tracks:

- AutoML: classical machine learning models trained with a shared `GridSearchCV` wrapper.
- AutoDL: PyTorch and TabNet based deep tabular baselines trained on the same feature matrix.

Quick run:

```bash
pip install -r requirements.txt
python main.py
python autodl/train.py
```

AutoML writes:

```text
outputs/automl_grid_search_results.json
```

AutoDL writes:

```text
outputs/autodl_training_results.json
```

## What is covered

- Quick start: dependencies, AutoML, AutoDL, Docker, and docs commands.
- Data and preprocessing: categorical columns, numeric columns, missing values, one-hot encoding, scaling, and passthrough columns.
- AutoML: shared grid search, metrics, refit strategy, and 13 model-specific grids.
- AutoDL: shared training loop, validation split, early stopping, and 5 model structures.
- Project notes: device selection, output files, repository layout, and PRDs.
