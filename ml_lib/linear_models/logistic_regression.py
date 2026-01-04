# ml_lib/linear_models/logistic_regression.py

import numpy as np

from ml_lib.core.base_model import BaseModel
from ml_lib.core.exceptions import NotFittedError


class LogisticRegression(BaseModel):
    """
    Binary Logistic Regression.
    Model: p = sigmoid(W·X + b)
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

        # Parameters to learn
        self.W = None
        self.b = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        train the LOGISTIC REGRESSION model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.W = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            # Linear part: z = W·X + b
            z = X @ self.W + self.b

            # Apply sigmoid
            y_pred = self._sigmoid(z)

            # Gradients
            dW = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.W -= self.lr * dW
            self.b -= self.lr * db

        return self

    def predict_proba(self, X):
        """
        Predict probability values.
        """
        if self.W is None:
            raise NotFittedError("Call fit() before predict_proba().")

        X = np.asarray(X)
        z = X @ self.W + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
