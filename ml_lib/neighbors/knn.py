# ml_lib/neighbors/knn.py

import numpy as np
from ml_lib.core.base_model import BaseModel


class KNN(BaseModel):
    """
    K-Nearest Neighbors algorithm (Classification & Regression).
    """

    def __init__(self, k=3, task="classification"):
        """
        Parameters:
             k : int (Number of nearest neighbors)
             task : str ('classification' or 'regression')
        """
        self.k = k
        self.task = task

        # Stored training data
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        (KNN has no training phase)
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def _euclidean_distance(self, x1, x2):
        """
        Compute Euclidean distance between two points.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_one(self, x):
        """
        Predict label/value for a single data point.
        """
        # Compute distance from x to all training points
        distances = []
        for x_train in self.X_train:
            dist = self._euclidean_distance(x, x_train)
            distances.append(dist)

        distances = np.array(distances)

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get corresponding labels
        k_labels = self.y_train[k_indices]

        # Classification or regression
        if self.task == "classification":
            # Majority vote
            values, counts = np.unique(k_labels, return_counts=True)
            return values[np.argmax(counts)]

        elif self.task == "regression":
            # Mean of neighbors
            return np.mean(k_labels)

        else:
            raise ValueError("task must be 'classification' or 'regression'")

    def predict(self, X):
        """
        Predict labels/values for given data.
        """
        X = np.asarray(X)

        predictions = []
        for x in X:
            pred = self._predict_one(x)
            predictions.append(pred)

        return np.array(predictions)
