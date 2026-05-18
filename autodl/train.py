from autodl.blstm import train_model as train_blstm
from autodl.cnn1d import train_model as train_cnn1d
from autodl.mlp import train_model as train_mlp
from autodl.tabnet import train_model as train_tabnet
from autodl.transformer import train_model as train_transformer
from preprocess import prepare_training_data
from utils.report_utils import print_result_summary, save_results

import config


MODEL_RUNNERS = {
    "mlp": train_mlp,
    "blstm": train_blstm,
    "cnn1d": train_cnn1d,
    "tabnet": train_tabnet,
    "transformer": train_transformer,
}


def main() -> None:
    """Run the unified AutoDL training entry."""
    # AutoDL 和 AutoML 共用同一份预处理结果，避免两套特征定义漂移。
    features, target = prepare_training_data()
    print(f"Prepared feature matrix shape: {features.shape}")
    print(f"Prepared target shape: {target.shape}")

    results = []
    for model_name in config.AUTODL_MODELS:
        runner = MODEL_RUNNERS[model_name]
        print(f"\nRunning deep model: {model_name}")
        result = runner(features, target)
        results.append(result)

    print_result_summary(results)
    save_results(results, config.AUTODL_RESULTS_PATH)


if __name__ == "__main__":
    main()
