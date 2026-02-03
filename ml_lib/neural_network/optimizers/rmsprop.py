import numpy as np
from ml_lib.core.optimizer import Optimizer


class RMSProp(Optimizer):

    def __init__(self, lr=0.001, beta=0.9,eps=1e-8):
        super().__init__(lr)
        self.beta=beta
        self.eps=eps
        self.s={}

    def update(self, params, grads):
        for key in params:
            if key not in self.s:
                self.s[key]=np.zeros_like(params[key])


            self.s[key]=self.beta*self.s[key]+(1-self.beta)*(grads[key]**2)
            params[key]-= self.lr*grads[key]/(np.sqrt(self.s[key])+ self.eps)
        