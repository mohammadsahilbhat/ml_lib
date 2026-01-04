import numpy as np 

class StandardScaler:
    def  __init__ (self):
        self.mean =None
        self.std =None  

    def fit(self,X):
        self.mean = np.mean(X,axis =0)
        self.std = np.std (X, axis =0)
        return self
    def transform (self,X):
        return(X-self.mean)/self.std + 1e-8
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
class MinMaxScaler:
    def  __init__(self):
        self.min =None
        self.max =None

    def fit (self,X):
        self.min = np.min (X, axis =0)
        self.max = np.max (X, axis =0)
        return self
    
    def transform (self,X):
        return (X -self.min)/(self.max -self.min + 1e-8)        

    def fit_transform (self,X):
        self.fit(X)
        return self.transform(X)