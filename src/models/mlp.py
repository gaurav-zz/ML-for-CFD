from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(
    hidden_layers: tuple[int, ...] = (128, 64),
    max_iter: int = 500,
    random_state: int = 42,
    *,
    batch_size: int | str = "auto",
    early_stopping: bool = True,
    learning_rate_init: float = 0.001,
) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=random_state,
            batch_size=batch_size,
            early_stopping=early_stopping,
            learning_rate_init=learning_rate_init,
        )),
    ])
