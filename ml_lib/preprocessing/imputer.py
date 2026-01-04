import numpy as np
import pandas as pd

class SimpleImputer:
    def __init__(self, strategy="mean", fill_value=None):
        """
        strategy: "mean", "median", "most_frequent", "constant"
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        self.statistics_ = np.zeros(n_features)

        for i in range(n_features):
            col = X[:, i]
            mask = ~np.isnan(col)

            if self.strategy == "mean":
                self.statistics_[i] = np.mean(col[mask])

            elif self.strategy == "median":
                self.statistics_[i] = np.median(col[mask])

            elif self.strategy == "most_frequent":
                values, counts = np.unique(col[mask], return_counts=True)
                self.statistics_[i] = values[np.argmax(counts)]

            elif self.strategy == "constant":
                self.statistics_[i] = self.fill_value

            else:
                raise ValueError("Invalid strategy")

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=float)

        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = self.statistics_[i]

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
