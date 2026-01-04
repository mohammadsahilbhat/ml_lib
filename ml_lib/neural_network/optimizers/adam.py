import numpy as np

class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        # Initialize momentums
        if layer not in self.m:
            self.m[layer] = {
                "dW": np.zeros_like(layer.W),
                "db": np.zeros_like(layer.b)
            }
            self.v[layer] = {
                "dW": np.zeros_like(layer.W),
                "db": np.zeros_like(layer.b)
            }

        self.t += 1
        m = self.m[layer]
        v = self.v[layer]

        # Update estimates
        m["dW"] = self.b1 * m["dW"] + (1 - self.b1) * layer.dW
        m["db"] = self.b1 * m["db"] + (1 - self.b1) * layer.db
        v["dW"] = self.b2 * v["dW"] + (1 - self.b2) * (layer.dW ** 2)
        v["db"] = self.b2 * v["db"] + (1 - self.b2) * (layer.db ** 2)

        # Bias correction
        m_hat_dw = m["dW"] / (1 - self.b1 ** self.t)
        m_hat_db = m["db"] / (1 - self.b2 ** self.t)
        v_hat_dw = v["dW"] / (1 - self.b2 ** self.t)
        v_hat_db = v["db"] / (1 - self.b2 ** self.t)

        # Parameter update
        layer.W -= self.lr * m_hat_dw / (np.sqrt(v_hat_dw) + self.eps)
        layer.b -= self.lr * m_hat_db / (np.sqrt(v_hat_db) + self.eps)

                 