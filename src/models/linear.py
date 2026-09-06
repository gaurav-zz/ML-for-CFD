from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
