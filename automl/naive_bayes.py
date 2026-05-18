from sklearn.naive_bayes import GaussianNB

from utils.train_utils import run_grid_search


def train_model(features, target):
    """Train Gaussian Naive Bayes with GridSearchCV."""
    # 朴素贝叶斯适合作为轻量基线模型保留在项目里。
    estimator = GaussianNB()
    param_grid = {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
    }
    return run_grid_search(
        model_name="naive_bayes",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
