"""
Preprocessing transformers and column transformers for leak-free pipelines.
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.pipeline import Pipeline as ImbPipeline

class SkewLog(BaseEstimator, TransformerMixin):
    """
    Applies signed log1p transformation: sign(x) * log1p(|x|) to numeric features
    whose absolute sample skewness exceeds the threshold on the training fold only.
    Prevents monetary attributes from dominating distance-based metrics and neural networks.
    """
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.skewed_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        sk = stats.skew(X_arr, axis=0, nan_policy="omit")
        self.skewed_ = np.abs(sk) > self.threshold
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float).copy()
        if self.skewed_ is not None and self.skewed_.any():
            X_arr[:, self.skewed_] = np.sign(X_arr[:, self.skewed_]) * np.log1p(np.abs(X_arr[:, self.skewed_]))
        return X_arr

def num_block():
    """Builds numerical pipeline: SkewLog followed by in-fold StandardScaler."""
    return ImbPipeline([
        ("skew", SkewLog(threshold=1.0)),
        ("scale", StandardScaler())
    ])

def pre_fixed(cat, num):
    """
    Corrected preprocessor:
    - Numerical: signed log1p + StandardScaler
    - Categorical: One-Hot Encoding
    """
    return ColumnTransformer([
        ("num", num_block(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
    ], sparse_threshold=0)

def pre_ordinal(cat, num, scale=False):
    """
    Replication preprocessor:
    - Categorical: OrdinalEncoder (reflecting paper's unstated default)
    - Numerical: optionally unscaled or scaled
    """
    return ColumnTransformer([
        ("num", num_block() if scale else "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)
    ], sparse_threshold=0)
