import numpy as np

from ._minipytorch import ActivationKind, FastLayer

__all__ = ["Module", "Linear", "Sequential"]


class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        for _name, _param in self.named_parameters():
            yield _param

    def named_parameters(self):
        return []

    def zero_grad(self):
        return self


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
            elif a in ("relu", "relu6"):
                kind = ActivationKind.RELU
            else:
                kind = ActivationKind.SIGMOID
        else:
            name = type(activation).__name__.lower()
            if "tanh" in name:
                kind = ActivationKind.TANH
            elif "relu" in name:
                kind = ActivationKind.RELU
            elif "sigmoid" in name:
                kind = ActivationKind.SIGMOID
            else:
                kind = ActivationKind.SIGMOID

        self.activation_kind = kind

        if weights is None:
            rng = np.random.default_rng(42)
            w = rng.standard_normal(size=(out_features * in_features), dtype=float) * 0.1
            weights_arr = w
        else:
            weights_arr = np.array(weights, dtype=float).reshape(-1)

        if bias is None:
            bias_arr = np.zeros(out_features, dtype=float)
        else:
            bias_arr = np.array(bias, dtype=float).reshape(-1)

        self._layer = FastLayer(
            num_inputs=in_features,
            num_outputs=out_features,
            activation_kind=kind.value if hasattr(kind, "value") else int(kind),
            weights=weights_arr,
            biases=bias_arr,
        )

    def forward(self, x):
        out = self._layer.forward(np.array(x, dtype=float))
        return out

    def backward(self, dloss_dout):
        return self._layer.backward(np.array(dloss_dout, dtype=float))

    def apply_gradients(self, lr=0.01):
        self._layer.apply_gradients(float(lr))
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
        yield "weight", self._layer.weights()
        yield "bias", self._layer.biases()


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out) if hasattr(layer, "__call__") else layer.forward(out)
        return out

    def named_parameters(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "named_parameters"):
                for name, param in layer.named_parameters():
                    yield f"{i}.{name}", param
            elif hasattr(layer, "parameters"):
                for j, param in enumerate(layer.parameters()):
                    yield f"{i}.param{j}", param

    def parameters(self):
        for _name, _param in self.named_parameters():
            yield _param
