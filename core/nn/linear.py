"""Fully-connected layer, backed by the native ``FastLayer``."""

import numpy as np

from .._minipytorch import ActivationKind, FastLayer
from .base import Module
from .params import Parameter

__all__ = ["Linear"]

_INIT_RNG = np.random.default_rng(42)

_ACTIVATION_NAMES = "'sigmoid', 'tanh', 'relu', 'relu6', or 'leaky_relu'"


class Linear(Module):
    def __init__(self, in_features, out_features, activation=None, weights=None, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Determine activation kind
        kind = ActivationKind.SIGMOID  # default
        if activation is None:
            kind = ActivationKind.SIGMOID
        elif isinstance(activation, ActivationKind):
            kind = activation
        elif isinstance(activation, str):
            a = activation.lower()
            if a == "sigmoid":
                kind = ActivationKind.SIGMOID
            elif a == "tanh":
                kind = ActivationKind.TANH
            elif a == "relu":
                kind = ActivationKind.RELU
            elif a == "relu6":
                kind = ActivationKind.RELU6
            elif a in ("leaky_relu", "leakyrelu", "leaky"):
                kind = ActivationKind.LEAKYRELU
            else:
                raise ValueError(
                    f"nn.Linear unknown activation {activation!r}; "
                    f"expected {_ACTIVATION_NAMES}"
                )
        else:
            name = type(activation).__name__.lower()
            if "tanh" in name:
                kind = ActivationKind.TANH
            elif "relu6" in name:
                kind = ActivationKind.RELU6
            elif "leaky" in name:
                kind = ActivationKind.LEAKYRELU
            elif "relu" in name:
                kind = ActivationKind.RELU
            elif "sigmoid" in name:
                kind = ActivationKind.SIGMOID
            else:
                raise ValueError(
                    f"nn.Linear unknown activation {activation!r} "
                    f"(type {type(activation).__name__}); "
                    f"expected None, an ActivationKind, one of {_ACTIVATION_NAMES}, "
                    f"or an activation module/class"
                )

        self.activation_kind = kind

        if not int(in_features) >= 0 or not int(out_features) >= 0:
            raise ValueError(
                f"nn.Linear needs non-negative in/out features, "
                f"got in={in_features}, out={out_features}"
            )

        if weights is None:
            w = _INIT_RNG.standard_normal(
                size=(out_features * in_features), dtype=float
            ) * 0.1
            weights_arr = w
        else:
            weights_arr = np.array(weights, dtype=float).reshape(-1)
            if weights_arr.size != in_features * out_features:
                raise ValueError(
                    f"nn.Linear expected {in_features * out_features} weight values "
                    f"({in_features} in x {out_features} out), got {weights_arr.size}"
                )

        if bias is None:
            bias_arr = np.zeros(out_features, dtype=float)
        else:
            bias_arr = np.array(bias, dtype=float).reshape(-1)
            if bias_arr.size != out_features:
                raise ValueError(
                    f"nn.Linear expected {out_features} bias values, "
                    f"got {bias_arr.size}"
                )

        self._layer = FastLayer(
            num_inputs=in_features,
            num_outputs=out_features,
            activation_kind=kind.value if hasattr(kind, "value") else int(kind),
            weights=weights_arr,
            biases=bias_arr,
        )

        self._weight_parameter = Parameter(self, "weight")
        self._bias_parameter = Parameter(self, "bias")
        self._has_forward = False

    def forward(self, x):
        x = np.array(x, dtype=float)
        # The native FastLayer reads num_inputs doubles straight out of the
        # buffer it is handed, so a short input is an out-of-bounds read: it
        # silently consumes whatever follows in memory, which shows up much
        # later as absurd gradients or a NaN loss. Reject the mismatch here.
        if x.size != self.in_features:
            raise ValueError(
                f"nn.Linear expected {self.in_features} input values, "
                f"got {x.size}"
            )
        out = self._layer.forward(x)
        self._has_forward = True
        return out

    def backward(self, dloss_dout):
        if not getattr(self, "_has_forward", False):
            raise RuntimeError(
                "nn.Linear.backward called before any forward"
            )
        dloss_dout = np.array(dloss_dout, dtype=float)
        if dloss_dout.size != self.out_features:
            raise ValueError(
                f"nn.Linear expected {self.out_features} gradient values, "
                f"got {dloss_dout.size}"
            )
        return self._layer.backward(dloss_dout)

    def apply_gradients(self, lr=0.01):
        self._layer.apply_gradients(float(lr))
        return self

    def zero_grad(self):
        self._layer.zero_gradients()
        return self

    def grad_weights(self):
        return self._layer.grad_weights()

    def grad_biases(self):
        return self._layer.grad_biases()

    def weights(self):
        return self._layer.weights()

    def biases(self):
        return self._layer.biases()

    def set_weights(self, w):
        self._layer.set_weights(np.array(w, dtype=float).reshape(-1))
        return self

    def set_biases(self, b):
        self._layer.set_biases(np.array(b, dtype=float).reshape(-1))
        return self

    @property
    def weight(self):
        return self._layer.weights()

    @property
    def bias(self):
        return self._layer.biases()

    def named_parameters(self):
        yield "weight", self._weight_parameter
        yield "bias", self._bias_parameter