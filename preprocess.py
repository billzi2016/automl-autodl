from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def load_training_frame(path: str | None = None) -> pd.DataFrame:
    """Load the training dataset from disk."""
    dataset_path = path or config.TRAIN_DATA_PATH
    return pd.read_csv(dataset_path)


def validate_feature_config(frame: pd.DataFrame) -> None:
    """Validate that configured columns exist in the dataset."""
    # 先校验配置里的列是否真实存在，避免训练跑起来后才发现字段拼错。
    required_columns = (
        [config.TARGET_COLUMN]
        + config.ONEHOT_COLUMNS
        + config.SCALE_COLUMNS
    )
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing configured columns: {missing_columns}")


def split_features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target."""
    features = frame.drop(columns=[config.TARGET_COLUMN]).copy()
    target = frame[config.TARGET_COLUMN].copy()
    return features, target


def build_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Apply missing value handling, one-hot encoding, and scaling."""
    # 类别列先补成统一占位符，再转成字符串，避免 one-hot 时出现类型混乱。
    categorical_frame = (
        features[config.ONEHOT_COLUMNS]
        .fillna(config.MISSING_CATEGORY_VALUE)
        .astype(str)
    )

    # 数值列统一转成数值格式，缺失值用中位数填充，更适合这类表格任务。
    scaled_source = features[config.SCALE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    scaled_source = scaled_source.fillna(scaled_source.median())
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(scaled_source)
    scaled_frame = pd.DataFrame(
        scaled_values,
        columns=config.SCALE_COLUMNS,
        index=features.index,
    )

    passthrough_columns = [
        column
        for column in features.columns
        if column not in config.ONEHOT_COLUMNS and column not in config.SCALE_COLUMNS
    ]
    passthrough_frame = features[passthrough_columns].copy()

    if not passthrough_frame.empty:
        # 其余列尽量保留下来，但也先转成模型可以直接消费的数值形式。
        for column in passthrough_frame.columns:
            passthrough_frame[column] = pd.to_numeric(
                passthrough_frame[column],
                errors="coerce",
            )
            passthrough_frame[column] = passthrough_frame[column].fillna(
                passthrough_frame[column].median()
            )

    encoded_frame = pd.get_dummies(
        categorical_frame,
        columns=config.ONEHOT_COLUMNS,
        dummy_na=False,
    )

    # 最终把标准化数值列、透传列和 one-hot 结果拼接成训练矩阵。
    processed_frame = pd.concat(
        [scaled_frame, passthrough_frame, encoded_frame],
        axis=1,
    )
    return processed_frame.astype(float)


def prepare_training_data(path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare the training dataset for downstream models."""
    # 入口函数只返回模型能直接消费的 X 和 y。
    frame = load_training_frame(path=path)
    validate_feature_config(frame)
    features, target = split_features_and_target(frame)
    processed_features = build_feature_matrix(features)
    return processed_features, target
