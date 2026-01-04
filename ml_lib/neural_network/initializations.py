import numpy as np 

def zeros(shape):
    return np.zeros(shape)

def random_normal(shape,scale=0.01):
    return np.random.randn(*shape)*scale

def xavier(shape):
    inp_dim  = shape[0]
    out_dim = shape[1]
    return np.random.randn(*shape)*np.sqrt(2.0/inp_dim + out_dim)
def he(shape):
    inp_dim  = shape[0]
    return np.random.randn(*shape)*np.sqrt(2.0/inp_dim)

def get_initializer(name):
    name = name.lower()
    if name == "zeros":
        return zeros
    elif name == "random_normal":
        return random_normal
    elif name == "xavier":
        return xavier
    elif name == "he":
        return he
    else:
        raise ValueError(f"Initializer '{name}' is not supported.")