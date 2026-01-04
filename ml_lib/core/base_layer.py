# ml_lib/core/base_layer.py

class Layer:
    """
    Base class for all neural network layers.
    """

    def forward(self, X):
        """
        Forward pass.
        """
        raise NotImplementedError("forward() not implemented")

    def backward(self, d_out):
        """
        Backward pass.
        """
        raise NotImplementedError("backward() not implemented")

    def params(self, lr):
        """
        Return trainable parameters.
        """
        return []

   