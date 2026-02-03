import numpy as np
from ml_lib.core.base_layer import Layer
from .initializations import get_initializer
from .get_activations import get_activation


class Dense(Layer):
    def __init__(self, units, init="xavier", activation=None):
        self.units = units
        self.init_name = init
        self.activation = get_activation(activation)
        self.W = None
        self.b = None
        self.initialized = False

    def _initialize(self, input_dim):
        init_fn = get_initializer(self.init_name)
        self.W = init_fn((input_dim, self.units))
        self.b = np.zeros(self.units)
        self.initialized = True

    def forward(self, X):
        if not self.initialized:
            self._initialize(X.shape[1])

        self.X = X
        Z = X @ self.W + self.b

        if self.activation:
            return self.activation.forward(Z)
        return Z

    def backward(self, d_out):
        if self.activation:
            d_out = self.activation.backward(d_out)

        if d_out.ndim == 1:
            d_out = d_out.reshape(-1, 1)  #  ensure 2D gradient

        self.dW = self.X.T @ d_out
        self.db = d_out.sum(axis=0)

        grad_input = d_out @ self.W.T

        return grad_input

    def params(self):
        return {"W": self.W, "b": self.b}
    
    def grads(self):
        return {"W": self.dW, "b": self.db}
