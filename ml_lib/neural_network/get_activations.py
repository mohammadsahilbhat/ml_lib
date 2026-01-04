from .activations import ReLU, Sigmoid,Tanh,Softmax



def get_activation(name):
    if name is None:
        return None

    name = name.lower()

    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "softmax":
        return Softmax()
    else:
        raise ValueError(f"Activation '{name}' is not supported")
