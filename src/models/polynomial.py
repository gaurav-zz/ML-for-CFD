from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_model(degree: int = 2) -> Pipeline:
    return Pipeline([("polynomial", PolynomialFeatures(degree=degree, include_bias=False)), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
