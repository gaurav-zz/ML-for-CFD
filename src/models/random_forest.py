from sklearn.ensemble import RandomForestRegressor


def build_model(n_estimators: int = 200, random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, min_samples_leaf=2)
