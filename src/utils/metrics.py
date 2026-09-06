"""Metrics reported for each aerodynamic target."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(actual, predicted)), "rmse": float(mean_squared_error(actual, predicted) ** 0.5), "r2": float(r2_score(actual, predicted))}
