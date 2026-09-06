from sklearn.ensemble import HistGradientBoostingRegressor


def build_model(random_state: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=random_state, max_iter=200, learning_rate=0.05, l2_regularization=1e-4)
