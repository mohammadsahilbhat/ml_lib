# ml_lib/linear_models/linear_regression.py

import numpy as np

from ml_lib.core.base_model import BaseModel
from ml_lib.core.exceptions import NotFittedError


class LinearRegression(BaseModel):
    """
    Linear Regression using Gradient Descent.
    Model: y = W·X + b
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

        # Parameters to learn
        self.W = None   # weights
        self.b = None   # bias

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.W = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            # Predictions
            y_pred = X @ self.W + self.b

            # Error
            error = y_pred - y

            # Gradients
            dj_dW = (1 / n_samples) * (X.T @ error)
            dj_db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.W -= self.lr * dj_dW
            self.b -= self.lr * dj_db

        return self

    def predict(self, X):
        """
        Predict output for given input.
        """
        if self.W is None:
            raise NotFittedError("Call fit() before predict().")

        X = np.asarray(X)
        return X @ self.W + self.b
