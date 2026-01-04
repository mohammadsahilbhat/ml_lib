import numpy as np 
from ml_lib.core.base_layer import Layer
# """
#     This scaler is intended ONLY for NN training
#     (batch-wise / internal normalization).
#     Do NOT use for dataset preprocessing.
#     """
class StandardScaler(Layer):
     
    def forward (self,X):
        self.mean =np.mean(X,axis=0)
        self.std= np.std(X,axis=0)
        return (X-self.mean)/self.std
    def backward(self, d_out):
        return d_out
    
class MinMaxScaler(Layer):
    def forward(self,X):
        self.min =np.min(X,axis=0)
        self.max=np.max(X,axis=0)
        return (X-self.min)/(self.max-self.min+1e-8)
    
    def backward (self,d_out):
        return d_out
    