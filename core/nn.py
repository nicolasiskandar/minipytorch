import numpy as np

from ._minipytorch import ActivationKind, FastLayer

__all__ = ["Module", "ModuleList", "Linear", "Sequential", "Softmax"]


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

    def children(self):
        for _name, _child in self.named_children():
            yield _child

    def named_children(self):
        return []

    def modules(self):
        yield self
        for child in self.children():
            if hasattr(child, "modules"):
                yield from child.modules()
            else:
                yield child

    def named_modules(self):
        yield "", self
        for name, child in self.named_children():
            if hasattr(child, "named_modules"):
                for cname, cmod in child.named_modules():
                    if cname:
                        yield f"{name}.{cname}", cmod
                    else:
                        yield name, cmod
            else:
                yield name, child

    def zero_grad(self):
        return self

    def train(self, mode=True):
        self._training = mode
        return self

    def eval(self):
        self._training = False
        return self

    def training(self):
        return getattr(self, "_training", True)


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
                kind = ActivationKind.SIGMOID
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

    def named_children(self):
        for i, layer in enumerate(self.layers):
            yield str(i), layer

    def children(self):
        for _name, child in self.named_children():
            yield child

    def named_parameters(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "named_parameters"):
                for name, param in layer.named_parameters():
                    if name:
                        yield f"{i}.{name}", param
                    else:
                        yield f"{i}", param
            elif hasattr(layer, "parameters"):
                for j, param in enumerate(layer.parameters()):
                    yield f"{i}.param{j}", param

    def parameters(self):
        for _name, _param in self.named_parameters():
            yield _param

    def named_modules(self):
        yield "", self
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "named_modules"):
                for name, mod in layer.named_modules():
                    if name:
                        yield f"{i}.{name}", mod
                    else:
                        yield f"{i}", mod
            else:
                yield str(i), layer

    def modules(self):
        yield self
        for _name, child in self.named_children():
            if hasattr(child, "modules"):
                yield from child.modules()
            else:
                yield child


class ModuleList(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = list(modules)

    def __iter__(self):
        return iter(self.modules_list)

    def __getitem__(self, idx):
        return self.modules_list[idx]

    def __len__(self):
        return len(self.modules_list)

    def append(self, module):
        self.modules_list.append(module)
        return self

    def named_children(self):
        for i, m in enumerate(self.modules_list):
            yield str(i), m

    def children(self):
        for _n, c in self.named_children():
            yield c

    def named_parameters(self):
        for i, m in enumerate(self.modules_list):
            if hasattr(m, "named_parameters"):
                for name, param in m.named_parameters():
                    if name:
                        yield f"{i}.{name}", param
                    else:
                        yield f"{i}", param
            elif hasattr(m, "parameters"):
                for j, param in enumerate(m.parameters()):
                    yield f"{i}.param{j}", param

    def parameters(self):
        for _n, p in self.named_parameters():
            yield p


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        from .activations import softmax

        x = np.asarray(x, dtype=float)
        dim = -1 if self.dim is None else self.dim
        if x.ndim > 0:
            dim = dim % x.ndim
        return softmax(x, dim=dim)
