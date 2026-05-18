# AutoML-AutoDL

This project is a unified tabular risk modeling workspace built around the Home Credit Default Risk dataset. It contains a shared preprocessing pipeline, a classical machine learning stack, and a deep learning stack for dense tabular classification experiments.

## Project Structure

```text
automl-autodl/
├── automl/          # Classical machine learning models with GridSearchCV
├── autodl/          # Deep learning tabular models built with PyTorch/TabNet
├── data/            # Sample dataset for demo and project structure display
├── utils/           # Shared training utilities, device logic, and reporting
├── config.py        # Unified configuration for preprocessing, AutoML, and AutoDL
├── preprocess.py    # Shared preprocessing pipeline
├── main.py          # AutoML entry
├── requirements.txt
└── README.md
```

## Data Pipeline

- Source dataset: `data/application_train.csv`
- Target column: `TARGET`
- Categorical features listed in `config.ONEHOT_COLUMNS`
- Numerical features listed in `config.SCALE_COLUMNS`
- Preprocessing steps:
  - missing value handling
  - one-hot encoding
  - numerical scaling
  - shared dense feature matrix generation

## AutoML Models

The `automl/` package contains one model per file for clear algorithm exposure:

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

Each AutoML model uses:

- shared `GridSearchCV`
- 3-fold cross validation
- `AUC`, `F1`, and `Accuracy` tracking
- unified result output format

## AutoDL Models

The `autodl/` package contains several deep tabular modeling baselines:

- `mlp.py`: multilayer perceptron baseline
- `blstm.py`: bidirectional LSTM over reshaped tabular sequences
- `cnn1d.py`: 1D convolutional classifier
- `tabnet.py`: TabNet-based tabular classifier
- `transformer.py`: transformer encoder for dense tabular inputs

The unified AutoDL entry is:

```bash
python autodl/train.py
```

## Device Strategy

Device detection is centralized in `utils/device_utils.py`.

- `CUDA` is preferred when an NVIDIA environment is available
- `MPS` is detected for Apple Silicon in PyTorch workflows
- unsupported GPU paths fall back to `CPU`
- XGBoost, LightGBM, and CatBoost only enable GPU parameters on compatible branches

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run AutoML:

```bash
python main.py
```

Run AutoDL:

```bash
python autodl/train.py
```

## Notes

- `data/application_train.csv` is kept in the repository as a sample dataset.
- Other generated CSV files under `data/` are ignored by `.gitignore`.
- Output result files are written to `outputs/`.
