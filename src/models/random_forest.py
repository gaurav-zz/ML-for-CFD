from sklearn.ensemble import RandomForestRegressor


def build_model(
    n_estimators: int = 200,
    random_state: int = 42,
    *,
    n_jobs: int = -1,
    min_samples_leaf: int = 2,
    max_depth: int | None = None,
    max_leaf_nodes: int | None = None,
    max_samples: int | float | None = None,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        max_samples=max_samples,
    )
