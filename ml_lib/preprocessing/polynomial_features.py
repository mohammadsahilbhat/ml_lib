import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=False):
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X):
        self.n_features = X.shape[1]
        return self

    def transform(self, X):
        # Start
        X_poly = []

        # Optional bias term
        if self.include_bias:
            X_poly.append(np.ones((X.shape[0], 1)))

        # Generate all degree combinations
        for deg in range(1, self.degree + 1):
            for comb in combinations_with_replacement(range(self.n_features), deg):
                X_poly.append(np.prod(X[:, comb], axis=1).reshape(-1, 1))

        # Concatenate column-wise
        return np.hstack(X_poly)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
  