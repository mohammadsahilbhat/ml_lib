import numpy as np
from ml_lib.core.optimizer import Optimizer

class Momentum(Optimizer):

    '''Stochastic Gradient Descent with Momentum'''
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta =beta
        self.v={}

    def update(self, params, grads):
        for key in params:
            if key not in self.v:
                self.v[key]=np.zeros_like(params[key])

            self.v[key]=self.beta*self.v[key]+(1-self.beta)*grads[key]
            params[key]-= self.lr*self.v[key]

    