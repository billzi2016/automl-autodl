# Data and preprocessing

The preprocessing entrypoint is `prepare_training_data()` in `preprocess.py`.

```python
features, target = prepare_training_data()
```

AutoML and AutoDL both use this same output, so their feature definitions do not drift apart.

## Input

Default file:

```text
data/application_train.csv
```

Target:

```text
TARGET
```

The relevant settings live in `config.py`:

- `TRAIN_DATA_PATH`
- `TARGET_COLUMN`
- `ONEHOT_COLUMNS`
- `SCALE_COLUMNS`

## Validation

`validate_feature_config()` checks that every configured column exists in the loaded dataframe. Missing columns raise a `ValueError` before training starts.

## Categorical columns

Categorical columns are filled with `Unknown`, converted to strings, and one-hot encoded with `pd.get_dummies()`.

```python
features[config.ONEHOT_COLUMNS]
    .fillna(config.MISSING_CATEGORY_VALUE)
    .astype(str)
```

## Numeric columns

Numeric columns are converted with `pd.to_numeric(errors="coerce")`, filled with their median, and scaled with `StandardScaler`.

Scaling matters for Logistic Regression, SVM, and KNN. Tree models are less sensitive to scale, but sharing one matrix keeps the project simple.

## Passthrough columns

Columns not listed in `ONEHOT_COLUMNS` or `SCALE_COLUMNS` are kept when they can be converted to numeric values. Missing values are filled with the median.

The final matrix concatenates:

- scaled numeric columns
- numeric passthrough columns
- one-hot encoded categorical columns

The result is cast to `float`, which works for scikit-learn, PyTorch, and TabNet.
