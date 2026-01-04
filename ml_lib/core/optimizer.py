class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        """
        params: dict -> {"W": W, "b": b}
        grads:  dict -> {"W": dW, "b": db}
        """
        raise NotImplementedError("Optimizer must implement update()")

