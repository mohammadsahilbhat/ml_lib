# ml_lib/core/base_model.py

class BaseModel:
    """
    Base class for all machine learning models.
    """

    def fit(self, X, y):
        """
        Train the model.
        """
        raise NotImplementedError("fit() not implemented")

    def predict(self, X):
        """
        Make predictions.
        """
        raise NotImplementedError("predict() not implemented")
