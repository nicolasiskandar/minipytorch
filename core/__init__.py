__version__ = "0.0.1"

from ._minipytorch import (
    ActivationKind,
    FastLayer,
    mse_loss,
    bce_loss,
    relu_deriv_from_output_f64,
    relu_f64,
    sigmoid_deriv_from_output_f64,
    sigmoid_f64,
    tanh_deriv_from_output_f64,
    tanh_f64,
)

from . import activations, losses, nn, optim
from .activations import *
from .losses import *
from .nn import *

__all__ = [
    "ActivationKind",
    "FastLayer",
    "mse_loss",
    "bce_loss",
    "sigmoid_f64",
    "tanh_f64",
    "relu_f64",
    "sigmoid_deriv_from_output_f64",
    "tanh_deriv_from_output_f64",
    "relu_deriv_from_output_f64",
    "activations",
    "losses",
    "nn",
    "Linear",
    "Sequential",
    "MSELoss",
    "BCELoss",
]

try:
    from .optim import *
except Exception:
    pass
from .optim import *

from .io import *
