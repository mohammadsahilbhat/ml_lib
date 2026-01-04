# ml_lib/core/exceptions.py

class NotFittedError(Exception):
    """
    Raised when predict() is called before fit().
    """
    pass
