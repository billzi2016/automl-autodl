from sklearn.neighbors import KNeighborsClassifier

from utils.train_utils import run_grid_search


def train_model(features, target):
    """Train KNN with GridSearchCV."""
    # KNN 作为距离度量模型，常用于对比线性模型和树模型效果。
    estimator = KNeighborsClassifier(
        n_jobs=1,
    )
    param_grid = {
        "n_neighbors": [5, 11, 21],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
    return run_grid_search(
        model_name="knn",
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        target=target,
    )
