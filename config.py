from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TRAIN_DATA_PATH = DATA_DIR / "application_train.csv"
AUTOML_RESULTS_PATH = OUTPUT_DIR / "automl_grid_search_results.json"
AUTODL_RESULTS_PATH = OUTPUT_DIR / "autodl_training_results.json"

TARGET_COLUMN = "TARGET"

# 统一在这里维护类别列，后续增删字段时不用到处改代码。
ONEHOT_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
]

# 这些数值列会做缺失值填充和标准化，方便线性模型与距离模型使用。
SCALE_COLUMNS = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

AUTOML_MODELS = [
    "logistic_regression",
    "svm",
    "knn",
    "naive_bayes",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "adaboost",
    "xgboost_model",
    "lightgbm_model",
    "catboost_model",
]

AUTODL_MODELS = [
    "mlp",
    "blstm",
    "cnn1d",
    "tabnet",
    "transformer",
]

CV_FOLDS = 3
GRIDSEARCH_N_JOBS = -1
RANDOM_STATE = 42
# 同时记录多个指标，但最终仍然用 AUC 选出最佳参数。
SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}
REFIT_METRIC = "roc_auc"
MISSING_CATEGORY_VALUE = "Unknown"

# 深度学习部分统一复用这一组配置，便于后续整体调参。
AUTODL_VALIDATION_SIZE = 0.2
AUTODL_BATCH_SIZE = 512
AUTODL_EPOCHS = 12
AUTODL_LEARNING_RATE = 1e-3
AUTODL_WEIGHT_DECAY = 1e-4
AUTODL_NUM_WORKERS = 0
AUTODL_HIDDEN_DIM = 128
AUTODL_DROPOUT = 0.2
AUTODL_EARLY_STOPPING_PATIENCE = 3
AUTODL_SEQUENCE_EMBED_DIM = 64
AUTODL_NUM_HEADS = 4
AUTODL_NUM_TRANSFORMER_LAYERS = 2

# 保留别名，避免入口脚本在迁移期间失效。
ENABLED_MODELS = AUTOML_MODELS
RESULTS_PATH = AUTOML_RESULTS_PATH
