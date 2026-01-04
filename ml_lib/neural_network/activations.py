import numpy as np
from ml_lib.core.base_layer import Layer

class ReLU(Layer):
    def forward(self, X):
        self.X=X
        return np.maximum(0,X)
    def backward (self, d_out):
        return d_out*(self.X>0)
    

class Sigmoid(Layer):
    def forward (self, X):
        self.out = 1/(1+np.exp(-X))
        return self.out
    def backward (self, d_out):
        return d_out*self.out*(1-self.out)
    
class Tanh(Layer):
    def forward (self, X):
        self.out = np.tanh(X)
        return self.out
    def backward (self, d_out):
        return d_out*(1-self.out**2)
    
class Softmax(Layer):
    def forward (self, X):
        exp = np.exp(X-np.max(X,axis=1,keepdims=True))
        self.out = exp/np.sum(exp,axis=1,keepdims=True)
        return self.out
    def backward (self, d_out):
        return d_out