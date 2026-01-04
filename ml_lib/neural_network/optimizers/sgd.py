import numpy as np
from ml_lib.core.optimizer import Optimizer

class SGD(Optimizer):

    def update(self, params, grads):
        for key in params:
            params[key]-=self.lr*grads[key]


