import numpy as np

from ._minipytorch import ActivationKind, FastLayer
from ._minipytorch import (
    conv2d_forward,
    conv2d_backward_input,
    conv2d_backward_weight,
    maxpool2d_forward,
    maxpool2d_backward,
    avgpool2d_forward,
    avgpool2d_backward,
)

__all__ = [
    "Module",
    "Parameter",
    "ModuleList",
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
]


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


def _pair(v):
    if isinstance(v, (tuple, list)):
        if len(v) != 2:
            raise ValueError(f"expected a pair of ints, got {v}")
        return (int(v[0]), int(v[1]))
    return (int(v), int(v))


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

        self._weight_parameter = Parameter(self, "weight")
        self._bias_parameter = Parameter(self, "bias")

    def forward(self, x):
        out = self._layer.forward(np.array(x, dtype=float))
        return out

    def backward(self, dloss_dout):
        return self._layer.backward(np.array(dloss_dout, dtype=float))

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
        from .activations import relu

        return relu(x)

    def _deriv_fn(self, y):
        from .activations import relu_deriv

        return relu_deriv(y)


class Sigmoid(_ElementwiseActivation):
    def __init__(self):
        super().__init__()
        self._init_state()

    def _forward_fn(self, x):
        from .activations import sigmoid

        return sigmoid(x)

    def _deriv_fn(self, y):
        from .activations import sigmoid_deriv

        return sigmoid_deriv(y)


class Tanh(_ElementwiseActivation):
    def __init__(self):
        super().__init__()
        self._init_state()

    def _forward_fn(self, x):
        from .activations import tanh

        return tanh(x)

    def _deriv_fn(self, y):
        from .activations import tanh_deriv

        return tanh_deriv(y)


class LeakyReLU(_ElementwiseActivation):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self._init_state()

    def _forward_fn(self, x):
        from .activations import leaky_relu

        return leaky_relu(x, alpha=self.negative_slope)

    def _deriv_fn(self, y):
        from .activations import leaky_relu_deriv

        return leaky_relu_deriv(y, alpha=self.negative_slope)


class ReLU6(_ElementwiseActivation):
    def __init__(self):
        super().__init__()
        self._init_state()

    def _forward_fn(self, x):
        from .activations import relu6

        return relu6(x)

    def _deriv_fn(self, y):
        from .activations import relu6_deriv

        return relu6_deriv(y)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        return x.reshape(-1)


class Conv2d(Module):
    """2D convolution over a single (unbatched) C x H x W input.

    Mirrors torch.nn.Conv2d's argument names but operates on one sample at a
    time, like Linear. Weights are stored as
    (out_channels, in_channels, kernel_height, kernel_width).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        weights=None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.use_bias = bool(bias)

        kh, kw = self.kernel_size
        shape = (self.out_channels, self.in_channels, kh, kw)
        if weights is None:
            rng = np.random.default_rng(0)
            w = rng.standard_normal(shape) * (1.0 / np.sqrt(self.in_channels * kh * kw))
        else:
            w = np.array(weights, dtype=float).reshape(shape)
        self._weights = w
        self._grad_weights = np.zeros(shape)
        self._bias = np.zeros(self.out_channels) if self.use_bias else np.zeros(0)
        self._grad_bias = np.zeros(self.out_channels) if self.use_bias else np.zeros(0)
        self._last_input = None
        self._last_input_shape = None

    def named_parameters(self):
        yield "weight", _ArrayParameter(self, "weight")
        if self.use_bias:
            yield "bias", _ArrayParameter(self, "bias")

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError(
                f"Conv2d expects a 3D (C, H, W) input, got shape {x.shape}"
            )
        c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Conv2d expected {self.in_channels} channels, got {c}"
            )
        kh, kw = self.kernel_size
        if h + 2 * self.padding[0] < kh or w + 2 * self.padding[1] < kw:
            raise ValueError(
                f"Conv2d input {h}x{w} is smaller than kernel {kh}x{kw}"
            )
        self._last_input = x.reshape(-1).copy()
        self._last_input_shape = (c, h, w)
        flat_out, out_shape = conv2d_forward(
            self._last_input,
            self._weights.reshape(-1),
            self._bias,
            c,
            h,
            w,
            self.out_channels,
            kh,
            kw,
            self.stride[0],
            self.padding[0],
        )
        return flat_out.reshape(out_shape)

    def backward(self, dloss_dout):
        dloss_dout = np.asarray(dloss_dout, dtype=float).reshape(-1)
        if self._last_input is None:
            raise RuntimeError("Conv2d.backward called before any forward")
        c, h, w = self._last_input_shape
        kh, kw = self.kernel_size
        grad_w, grad_b = conv2d_backward_weight(
            dloss_dout,
            self._last_input,
            c,
            h,
            w,
            self.out_channels,
            kh,
            kw,
            self.stride[0],
            self.padding[0],
        )
        self._grad_weights = grad_w.reshape(self._weights.shape)
        self._grad_bias = grad_b
        grad_in = conv2d_backward_input(
            dloss_dout,
            self._weights.reshape(-1),
            c,
            h,
            w,
            self.out_channels,
            kh,
            kw,
            self.stride[0],
            self.padding[0],
        )
        return grad_in.reshape(self._last_input_shape)

    def zero_grad(self):
        self._grad_weights = np.zeros_like(self._weights)
        self._grad_bias = np.zeros_like(self._bias)
        return self

    def apply_gradients(self, lr=0.01):
        self._weights = self._weights - lr * self._grad_weights
        if self.use_bias:
            self._bias = self._bias - lr * self._grad_bias
        return self


class MaxPool2d(Module):
    """2D max pooling over a single (unbatched) C x H x W input."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            self.stride = _pair(kernel_size)
        else:
            self.stride = _pair(stride)
        self.padding = _pair(padding)
        self._last_shape = None
        self._last_out_shape = None
        self._indices = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError(
                f"MaxPool2d expects a 3D (C, H, W) input, got shape {x.shape}"
            )
        c, h, w = x.shape
        if h + 2 * self.padding[0] < self.kernel_size[0] or w + 2 * self.padding[1] < self.kernel_size[1]:
            raise ValueError(
                f"MaxPool2d input {h}x{w} is smaller than kernel {self.kernel_size}"
            )
        out, indices, out_shape = maxpool2d_forward(
            x.reshape(-1),
            c,
            h,
            w,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
        )
        self._last_shape = (c, h, w)
        self._last_out_shape = tuple(int(v) for v in out_shape)
        self._indices = indices
        return out.reshape(self._last_out_shape)

    def backward(self, dloss_dout):
        dloss_dout = np.asarray(dloss_dout, dtype=float).reshape(-1)
        if self._indices is None:
            raise RuntimeError("MaxPool2d.backward called before any forward")
        c, h, w = self._last_shape
        oc, oh, ow = self._last_out_shape
        grad_in = maxpool2d_backward(
            dloss_dout, self._indices, c, h, w, oc, oh, ow
        )
        return grad_in.reshape(c, h, w)


class AvgPool2d(Module):
    """2D average pooling over a single (unbatched) C x H x W input."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            self.stride = _pair(kernel_size)
        else:
            self.stride = _pair(stride)
        self.padding = _pair(padding)
        self._last_shape = None
        self._last_out_shape = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError(
                f"AvgPool2d expects a 3D (C, H, W) input, got shape {x.shape}"
            )
        c, h, w = x.shape
        if h + 2 * self.padding[0] < self.kernel_size[0] or w + 2 * self.padding[1] < self.kernel_size[1]:
            raise ValueError(
                f"AvgPool2d input {h}x{w} is smaller than kernel {self.kernel_size}"
            )
        out, out_shape = avgpool2d_forward(
            x.reshape(-1),
            c,
            h,
            w,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
        )
        self._last_shape = (c, h, w)
        self._last_out_shape = tuple(int(v) for v in out_shape)
        return out.reshape(self._last_out_shape)

    def backward(self, dloss_dout):
        dloss_dout = np.asarray(dloss_dout, dtype=float).reshape(-1)
        if self._last_shape is None:
            raise RuntimeError("AvgPool2d.backward called before any forward")
        c, h, w = self._last_shape
        oc, oh, ow = self._last_out_shape
        grad_in = avgpool2d_backward(
            dloss_dout,
            c,
            h,
            w,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
            oc,
            oh,
            ow,
        )
        return grad_in.reshape(c, h, w)
