import numpy as np 
from ml_lib.core.loss import Loss


class MSE(Loss):
    def forward (self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true)**2)
    def backward(self):
        grad = 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
        return grad.reshape(-1, 1) 
    
class BinaryCrossEntropy(Loss):
    def forward (self, y_pred, y_true):
        ep=1e-8
        self.y_pred = np.clip (y_pred , ep ,1-ep)
        self.y_true = y_true
        loss = -(y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        return np.mean(loss)
    def backward(self):
        n = self.y_true.shape[0]
        ep=1e-8
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * n + ep)
    


class CategoricalCrossEntropy(Loss):
    

    def forward(self, y_pred, y_true):
        eps = 1e-8
        self.y_pred = np.clip(y_pred, eps, 1 - eps)
        self.y_true = y_true

        loss = -np.sum(y_true * np.log(self.y_pred), axis=1)
        return np.mean(loss)

    def backward(self):
        return self.y_pred - self.y_true