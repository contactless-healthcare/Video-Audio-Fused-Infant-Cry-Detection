"""Missing-value preprocessing used by late-fusion base models."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class FixedMissingIndicatorImputer(BaseEstimator, TransformerMixin):
    """Median-impute values and append fixed NaN indicators."""

    def __init__(self, indicator_indices):
        self.indicator_indices = indicator_indices

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        indicators = np.isnan(values[:, self.indicator_indices]).astype(float)
        return np.hstack([self.imputer_.transform(values), indicators])
