from automl.adaboost import train_model as train_adaboost
from automl.catboost_model import train_model as train_catboost_model
from automl.decision_tree import train_model as train_decision_tree
from automl.extra_trees import train_model as train_extra_trees
from automl.gradient_boosting import train_model as train_gradient_boosting
from automl.hist_gradient_boosting import train_model as train_hist_gradient_boosting
from automl.knn import train_model as train_knn
from automl.lightgbm_model import train_model as train_lightgbm_model
from automl.logistic_regression import train_model as train_logistic_regression
from automl.naive_bayes import train_model as train_naive_bayes
from automl.random_forest import train_model as train_random_forest
from automl.svm import train_model as train_svm
from automl.xgboost_model import train_model as train_xgboost_model
from preprocess import prepare_training_data
from utils.report_utils import print_result_summary, save_results

import config


MODEL_RUNNERS = {
    "logistic_regression": train_logistic_regression,
    "svm": train_svm,
    "knn": train_knn,
    "naive_bayes": train_naive_bayes,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "extra_trees": train_extra_trees,
    "gradient_boosting": train_gradient_boosting,
    "hist_gradient_boosting": train_hist_gradient_boosting,
    "adaboost": train_adaboost,
    "xgboost_model": train_xgboost_model,
    "lightgbm_model": train_lightgbm_model,
    "catboost_model": train_catboost_model,
}


def main() -> None:
    """Run the preprocessing pipeline and model searches."""
    # 先统一完成预处理，后续所有模型复用同一份特征矩阵。
    features, target = prepare_training_data()
    print(f"Prepared feature matrix shape: {features.shape}")
    print(f"Prepared target shape: {target.shape}")

    results = []

    for model_name in config.ENABLED_MODELS:
        # 每个模型的训练逻辑拆到独立文件，入口层只做调度和结果汇总。
        runner = MODEL_RUNNERS[model_name]
        print(f"\nRunning model: {model_name}")
        result = runner(features, target)
        results.append(result)

    print_result_summary(results)
    save_results(results, config.RESULTS_PATH)


if __name__ == "__main__":
    main()
