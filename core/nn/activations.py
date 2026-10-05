"""Activation *modules*.

Note the relative-import level: the functional kernels live in
``core.activations``, which is a sibling of this package, not of this module.
Because this file is itself called ``activations.py``, the kernels must be
reached with two dots -- ``from .activations import relu`` would import this
file. The imports stay inside the methods so that ``core.nn`` never holds
``core.activations`` open while it is still initialising.
"""

import numpy as np

from .base import Module

__all__ = [
    "LeakyReLU",
    "ReLU",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Tanh",
]


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        from ..activations import softmax

        x = np.asarray(x, dtype=float)
        dim = -1 if self.dim is None else self.dim
        if x.ndim > 0:
            dim = dim % x.ndim
        return softmax(x, dim=dim)


class _ElementwiseActivation(Module):
    """Shared plumbing for pointwise activations.

    Each subclass names the forward and derivative functions. The derivative is
    taken from the *output* (all five kernels expose deriv-from-output forms),
    so the output has to be kept from forward for backward to use.
    """

    _forward_fn = None
    _deriv_fn = None

    def forward(self, x):
        out = np.asarray(self._forward_fn(x), dtype=float)
        self._last_output = out
        return out

    def backward(self, dloss_dout):
        if self._last_output is None:
            raise RuntimeError(
                f"{type(self).__name__}.backward called before any forward"
            )
        dloss_dout = np.asarray(dloss_dout, dtype=float)
        return dloss_dout * np.asarray(self._deriv_fn(self._last_output), dtype=float)

    def _init_state(self):
        self._last_output = None


class ReLU(_ElementwiseActivation):
    def __init__(self):
        super().__init__()
        self._init_state()

    def _forward_fn(self, x):
        from ..activations import relu

        return relu(x)

    def _deriv_fn(self, y):
        from ..activations import relu_deriv

        return relu_deriv(y)


class Sigmoid(_ElementwiseActivation):
    def __init__(self):
        super().__init__()
        self._init_state()

    def _forward_fn(self, x):
        from ..activations import sigmoid

        return sigmoid(x)

    def _deriv_fn(self, y):
        from ..activations import sigmoid_deriv

        return sigmoid_deriv(y)


class Tanh(_ElementwiseActivation):
    def __init__(self):
        super().__init__()
        self._init_state()

    def _forward_fn(self, x):
        from ..activations import tanh

        return tanh(x)

    def _deriv_fn(self, y):
        from ..activations import tanh_deriv

        return tanh_deriv(y)


class LeakyReLU(_ElementwiseActivation):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self._init_state()

    def _forward_fn(self, x):
        from ..activations import leaky_relu

        return leaky_relu(x, alpha=self.negative_slope)

    def _deriv_fn(self, y):
        from ..activations import leaky_relu_deriv

        return leaky_relu_deriv(y, alpha=self.negative_slope)


class ReLU6(_ElementwiseActivation):
    def __init__(self):
        super().__init__()
        self._init_state()

    def _forward_fn(self, x):
        from ..activations import relu6

        return relu6(x)

    def _deriv_fn(self, y):
        from ..activations import relu6_deriv

        return relu6_deriv(y)