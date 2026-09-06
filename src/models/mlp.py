from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(hidden_layers: tuple[int, ...] = (128, 64), max_iter: int = 500, random_state: int = 42) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=max_iter, early_stopping=True, random_state=random_state))])
