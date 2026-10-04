from .core import *
from .core import activations, losses, nn, optim
from .core.activations import ActivationKind
from .core.losses import MSELoss, BCELoss, CrossEntropyLoss, mse_loss, bce_loss
from .core.nn import Linear, Sequential, ModuleList, Softmax

__version__ = "0.0.1"

__all__ = [
    "Linear",
    "Sequential",
    "ModuleList",
    "Softmax",
    "MSELoss",
    "BCELoss",
    "CrossEntropyLoss",
    "bce_loss",
    "mse_loss",
    "ActivationKind",
    "activations",
    "losses",
    "nn",
]

try:
    from .core.optim import SGD
except Exception:
    pass
