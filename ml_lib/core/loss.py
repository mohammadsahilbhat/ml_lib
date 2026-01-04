# ml_lib/core/loss.py

class Loss:
    """
    Base class for loss functions.
    """

    def forward(self, y_pred, y_true):
        """
        Compute loss value.
        """
        raise NotImplementedError("forward() not implemented")

    def backward(self):
        """
        Compute gradient w.r.t predictions.
        """
        raise NotImplementedError("backward() not implemented")
