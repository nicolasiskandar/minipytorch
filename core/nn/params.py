"""Parameter handles handed to optimizers.

Two implementations with one surface, because two kinds of layer own their
storage differently:

* :class:`Parameter` talks to a native ``FastLayer`` through its methods.
* :class:`_ArrayParameter` talks to a layer that owns plain numpy arrays.

Both own their momentum buffer and write updates back through the owning layer,
so repeated ``parameters()`` calls must return the *same* objects or momentum
resets every step. ``optim.SGD`` rejects anything that is not one of these.
"""

import numpy as np

__all__ = ["Parameter"]


class Parameter:
    def __init__(self, layer, kind):
        if kind not in ("weight", "bias"):
            raise ValueError(f"unknown parameter kind: {kind}")
        self._layer = layer
        self._kind = kind
        self._velocity = None

    @property
    def layer(self):
        return self._layer

    @property
    def kind(self):
        return self._kind

    @property
    def value(self):
        if self._kind == "weight":
            return self._layer.weights()
        return self._layer.biases()

    def grad(self):
        if self._kind == "weight":
            return self._layer.grad_weights()
        return self._layer.grad_biases()

    def set_value(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if self._kind == "weight":
            self._layer.set_weights(values)
        else:
            self._layer.set_biases(values)
        return self

    def zero_grad(self):
        self._layer.zero_grad()

    def apply_gradients(self, lr=0.01, momentum=0.0):
        gradients = np.asarray(self.grad(), dtype=float)
        values = np.asarray(self.value, dtype=float)
        if momentum == 0.0:
            updated = values - lr * gradients
        else:
            if self._velocity is None or self._velocity.shape != gradients.shape:
                self._velocity = np.zeros_like(gradients)
            self._velocity = momentum * self._velocity + gradients
            updated = values - lr * self._velocity
        if self._kind == "weight":
            self._layer.set_weights(updated)
        else:
            self._layer.set_biases(updated)
        return self


class _ArrayParameter:
    """Parameter handle for layers that own plain numpy arrays.

    Mirrors the FastLayer-backed Parameter surface (layer/kind/value/grad/
    zero_grad/apply_gradients) so optimizers cannot tell the difference.
    """

    def __init__(self, layer, kind):
        if kind not in ("weight", "bias"):
            raise ValueError(f"unknown parameter kind: {kind}")
        self._layer = layer
        self._kind = kind
        self._velocity = None

    @property
    def layer(self):
        return self._layer

    @property
    def kind(self):
        return self._kind

    @property
    def value(self):
        return self._layer._weights if self._kind == "weight" else self._layer._bias

    def grad(self):
        return (
            self._layer._grad_weights if self._kind == "weight" else self._layer._grad_bias
        )

    def set_value(self, values):
        values = np.asarray(values, dtype=float)
        if self._kind == "weight":
            self._layer._weights = values
        else:
            self._layer._bias = values
        return self

    def zero_grad(self):
        return self._layer.zero_grad()

    def apply_gradients(self, lr=0.01, momentum=0.0):
        gradients = np.asarray(self.grad(), dtype=float)
        values = np.asarray(self.value, dtype=float)
        if momentum == 0.0:
            updated = values - lr * gradients
        else:
            if self._velocity is None or self._velocity.shape != gradients.shape:
                self._velocity = np.zeros_like(gradients)
            self._velocity = momentum * self._velocity + gradients
            updated = values - lr * self._velocity
        if self._kind == "weight":
            self._layer._weights = updated
        else:
            self._layer._bias = updated
        return self