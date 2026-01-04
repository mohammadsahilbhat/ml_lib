from ml_lib.neural_network.optimizers.adam import Adam
from ml_lib.neural_network.optimizers.sgd import SGD
from ml_lib.neural_network.optimizers.momentum import Momentum
from ml_lib.neural_network.optimizers.rmsprop import RMSProp


def get_optimizer(optimizer, lr=0.001):
    """
    Returns an optimizer instance.
    Accepts:
    - optimizer instance → returned directly
    - optimizer name (string) → mapped to correct class
    """

    # If user passed already-created optimizer instance
    if hasattr(optimizer, "update"):
        return optimizer

    # If string → convert to optimizer object
    if isinstance(optimizer, str):
        opt = optimizer.lower()

        if opt == "adam":
            return Adam(lr=lr)

        elif opt == "sgd":
            return SGD(lr=lr)

        elif opt in ["momentum", "sgd_momentum", "msgd"]:
            return Momentum(lr=lr)

        elif opt == "rmsprop":
            return RMSProp(lr=lr)

        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'")

    raise TypeError("Optimizer must be an optimizer instance or string name.")