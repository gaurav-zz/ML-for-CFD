from sklearn.ensemble import HistGradientBoostingRegressor


def build_model(
    random_state: int = 42,
    *,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    early_stopping: bool | str = "auto",
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        early_stopping=early_stopping,
        l2_regularization=1e-4,
    )
