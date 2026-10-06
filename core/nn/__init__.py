"""Layer definitions and the :class:`Module` base."""

from . import functional
from .base import Module
from .params import Parameter, _ArrayParameter
from ._shape import _pair
from .linear import Linear
from .containers import Sequential
from .reshape import Flatten
from .conv import Conv2d
from .pool import AvgPool2d, MaxPool2d
from .dropout import Dropout
from .activations import (
    LeakyReLU,
    ReLU,
    ReLU6,
    Sigmoid,
    Softmax,
    Tanh,
    _ElementwiseActivation,
)

from .._minipytorch import ActivationKind, FastLayer
from .._minipytorch import (
    avgpool2d_backward,
    avgpool2d_forward,
    conv2d_backward_input,
    conv2d_backward_weight,
    conv2d_forward,
    maxpool2d_backward,
    maxpool2d_forward,
)

__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "Sequential",
    "Softmax",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "ReLU6",
    "Flatten",
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
    "Dropout",
]