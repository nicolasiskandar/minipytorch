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
    """Softmax over ``dim``.

    ``backward`` needs the forward output. The Jacobian is
    ``J = diag(y) - y y^T``, so for an incoming gradient ``g``:

        grad = J.T @ g = diag(y) @ g - y * dot(y, g)
             = y * (g - dot(y, g))

    Caching ``y`` is what makes this cheap -- same trick as the elementwise
    activations, whose kernels also take a deriv-from-output form.

    Note for the common classifier: ``CrossEntropyLoss`` already applies softmax
    to its input and returns a gradient with respect to the *logits*. Pairing it
    with this layer would apply softmax twice. The intended composition is
    ``Linear -> CrossEntropyLoss``.
    """

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim
        self._last_output = None

    def _resolve_dim(self, ndim):
        dim = -1 if self.dim is None else self.dim
        return dim % ndim if ndim > 0 else 0

    def forward(self, x):
        from ..activations import softmax

        x = np.asarray(x, dtype=float)
        out = softmax(x, dim=self._resolve_dim(x.ndim))
        self._last_output = out.copy()
        return out

    def backward(self, dloss_dout):
        if self._last_output is None:
            raise RuntimeError(
                "Softmax.backward called before any forward"
            )
        y = self._last_output
        g = np.asarray(dloss_dout, dtype=float)
        # Every reduction runs over the normalised axis only, so a (N, C) batch
        # is reduced per row rather than collapsing the whole batch into one
        # scalar -- which would mix samples together.
        axis = self._resolve_dim(y.ndim)
        return y * (g - np.sum(y * g, axis=axis, keepdims=True))


class _ElementwiseActivation(Module):
    """Shared plumbing for pointwise activations.

    Each subclass names the forward and derivative functions. The derivative is
    taken from the *output* (all five kernels expose deriv-from-output forms),
    so the output has to be kept from forward for backward to use.
    """

    _forward_fn = None
    _deriv_fn = None

    def forward(self, x):
        out = np.array(self._forward_fn(x), dtype=float)
        # Cache a private copy: backward derives from the activation *output*,
        # so if the caller mutates the returned array the derivative must not
        # silently see the mutated values. Returning and caching the very same
        # buffer made a user doing y[i] = v corrupt the cached derivative.
        self._last_output = out.copy()
        return out

    def backward(self, dloss_dout):
        if self._last_output is None:
            raise RuntimeError(
                f"{type(self).__name__}.backward called before any forward"
            )
        dloss_dout = np.asarray(dloss_dout, dtype=float)
        y = self._last_output
        if dloss_dout.shape != y.shape:
            if dloss_dout.size != y.size:
                raise ValueError(
                    f"{type(self).__name__}.backward expected {y.size} "
                    f"gradient values matching the {y.shape} activation, "
                    f"got {dloss_dout.size} in shape {dloss_dout.shape}"
                )
            dloss_dout = dloss_dout.reshape(y.shape)
        return dloss_dout * np.asarray(self._deriv_fn(y), dtype=float)

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
        if inplace:
            raise NotImplementedError(
                "LeakyReLU(inplace=True) is not supported; the layer never "
                "mutates its input, so pass inplace=False"
            )
        self.negative_slope = negative_slope
        self.inplace = False
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