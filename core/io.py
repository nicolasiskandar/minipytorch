__all__ = ["save_model", "load_model"]

import numpy as np

from .nn import (
    AvgPool2d,
    Conv2d,
    Flatten,
    LeakyReLU,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    ReLU6,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
)
from .activations import ActivationKind

_ACTIVATION_NAMES = {
    int(ActivationKind.SIGMOID): "sigmoid",
    int(ActivationKind.TANH): "tanh",
    int(ActivationKind.RELU): "relu",
    int(ActivationKind.RELU6): "relu6",
    int(ActivationKind.LEAKYRELU): "leaky_relu",
}

_REVERSE_ACTIVATION_NAMES = {
    "sigmoid": ActivationKind.SIGMOID,
    "tanh": ActivationKind.TANH,
    "relu": ActivationKind.RELU,
    "relu6": ActivationKind.RELU6,
    "leaky_relu": ActivationKind.LEAKYRELU,
}

_FORMAT_TAG = "minipytorch-v2"


def _get_activation_name(kind: ActivationKind) -> str:
    name = _ACTIVATION_NAMES.get(int(kind))
    if name is None:
        raise ValueError(f"no save format for activation kind {int(kind)}")
    return name


_STATELESS_LAYERS = {
    "relu": (ReLU, ()),
    "sigmoid": (Sigmoid, ()),
    "tanh": (Tanh, ()),
    "relu6": (ReLU6, ()),
    "softmax": (lambda: Softmax(), ()),
    "flatten": (Flatten, ("start_dim",)),
    "maxpool2d": (MaxPool2d, ("kernel_size", "stride", "padding")),
    "avgpool2d": (AvgPool2d, ("kernel_size", "stride", "padding")),
    "leaky_relu": (LeakyReLU, ("negative_slope",)),
}


def _is_legacy_linear_only(layers):
    return all(isinstance(layer, Linear) for layer in layers)


def _fmt(values):
    return " ".join(f"{float(v):.17g}" for v in values)


def _save_tagged(out, layers):
    out.write(f"{_FORMAT_TAG}\n")
    out.write(f"{len(layers)}\n")
    for layer in layers:
        if isinstance(layer, Linear):
            out.write(
                f"linear {layer.in_features} {layer.out_features} "
                f"{_get_activation_name(layer.activation_kind)}\n"
            )
            out.write(f"{layer.out_features}\n")
            w = np.array(layer.weights(), dtype=float).reshape(
                layer.out_features, layer.in_features
            )
            b = np.array(layer.biases(), dtype=float)
            for i in range(layer.out_features):
                bias = float(b[i]) if i < len(b) else 0.0
                out.write(f"{_fmt([bias])} {_fmt(w[i])}\n")
        elif isinstance(layer, Conv2d):
            kh, kw = layer.kernel_size
            sh, sw = layer.stride
            ph, pw = layer.padding
            out.write(
                f"conv2d {layer.in_channels} {layer.out_channels} {kh} {kw} "
                f"{sh} {ph} {int(layer.use_bias)}\n"
            )
            weight_count = layer._weights.size
            bias_count = layer._bias.size
            out.write(f"{weight_count} {bias_count}\n")
            out.write(_fmt(layer._weights.reshape(-1)) + "\n")
            if bias_count:
                out.write(_fmt(layer._bias.reshape(-1)) + "\n")
        else:
            tag = None
            for name, (factory, _) in _STATELESS_LAYERS.items():
                if type(layer) is factory:
                    tag = name
                    break
            if tag is None:
                raise TypeError(
                    "save_model has no format for layer type "
                    f"{type(layer).__name__}"
                )
            args = []
            if tag in ("maxpool2d", "avgpool2d"):
                args = [
                    str(int(layer.kernel_size[0])),
                    str(int(layer.stride[0])),
                    str(int(layer.padding[0])),
                ]
            elif tag == "leaky_relu":
                args = [repr(float(layer.negative_slope))]
            elif tag == "flatten":
                args = [str(int(layer.start_dim))]
            out.write(f"{tag} {' '.join(args)}\n")


def save_model(model, filename: str) -> None:
    if isinstance(model, Sequential):
        layers = list(model.layers)
    else:
        layers = [model]

    with open(filename, "w", encoding="utf-8") as out:
        if _is_legacy_linear_only(layers):
            _save_legacy(out, layers)
        else:
            _save_tagged(out, layers)


def _save_legacy(out, layers):
    """Original positional format, kept byte-for-byte for Linear-only nets."""
    out.write(f"{len(layers)}\n")
    for layer in layers:
        if not isinstance(layer, Linear):
            raise TypeError("save_model only supports Sequential of Linear layers")
        w = np.array(layer.weights(), dtype=float).reshape(
            layer.out_features, layer.in_features
        )
        b = np.array(layer.biases(), dtype=float)
        out.write(f"{layer.out_features}\n")
        act_name = _get_activation_name(layer.activation_kind)
        for i in range(layer.out_features):
            row = w[i]
            bias = float(b[i]) if i < len(b) else 0.0
            parts = [act_name, f"{bias:.17g}"]
            for j in range(layer.in_features):
                parts.append(f"{float(row[j]):.17g}")
            out.write(" ".join(parts) + "\n")


def _next_nonblank(lines, idx):
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return idx


def load_model(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = _next_nonblank(lines, 0)
    if idx >= len(lines):
        raise ValueError(f"{filename!r} is empty")
    first = lines[idx].strip()
    if first == _FORMAT_TAG:
        return _load_tagged(lines, idx + 1, filename)
    return _load_legacy(lines, idx, filename)


def _load_legacy(lines, idx, filename):
    num_layers = int(lines[idx].strip())
    idx += 1

    layers = []
    for _ in range(num_layers):
        idx = _next_nonblank(lines, idx)
        num_neurons = int(lines[idx].strip())
        idx += 1
        weights_list = []
        biases_list = []
        act_kind = ActivationKind.SIGMOID
        in_features = 0
        for i in range(num_neurons):
            line = lines[idx].rstrip()
            idx += 1
            parts = line.split()
            if len(parts) == 0:
                i -= 1
                continue
            act_name = parts[0]
            if act_name not in _REVERSE_ACTIVATION_NAMES:
                raise ValueError(
                    f"unknown activation {act_name!r} on line {idx} of {filename!r}"
                )
            act_kind = _REVERSE_ACTIVATION_NAMES[act_name]
            bias_val = float(parts[1])
            wparts = parts[2:] if len(parts) > 2 else []
            weights = [float(p) for p in wparts]
            if in_features == 0 and len(weights) > 0:
                in_features = len(weights)
            weights_list.extend(weights)
            biases_list.append(bias_val)
        out_features = num_neurons
        if in_features == 0:
            in_features = 1
        layers.append(
            Linear(
                in_features=in_features,
                out_features=out_features,
                activation=act_kind,
                weights=weights_list,
                bias=biases_list,
            )
        )

    return Sequential(*layers)


def _load_tagged(lines, idx, filename):
    idx = _next_nonblank(lines, idx)
    num_layers = int(lines[idx].strip())
    idx += 1

    layers = []
    for _ in range(num_layers):
        idx = _next_nonblank(lines, idx)
        parts = lines[idx].split()
        idx += 1
        if not parts:
            raise ValueError(f"malformed layer header in {filename!r}")
        tag = parts[0]

        if tag == "linear":
            if len(parts) < 4:
                raise ValueError(f"malformed linear layer in {filename!r}")
            in_features = int(parts[1])
            out_features = int(parts[2])
            act_name = parts[3]
            if act_name not in _REVERSE_ACTIVATION_NAMES:
                raise ValueError(
                    f"unknown activation {act_name!r} in {filename!r}"
                )
            act_kind = _REVERSE_ACTIVATION_NAMES[act_name]
            idx = _next_nonblank(lines, idx)
            count = int(lines[idx].strip())
            idx += 1
            weights, biases = [], []
            for _ in range(count):
                fields = lines[idx].split()
                idx += 1
                if len(fields) < 1 + in_features:
                    raise ValueError(f"truncated linear weights in {filename!r}")
                biases.append(float(fields[0]))
                weights.extend(float(v) for v in fields[1 : 1 + in_features])
            layers.append(
                Linear(
                    in_features=in_features,
                    out_features=out_features,
                    activation=act_kind,
                    weights=weights,
                    bias=biases,
                )
            )
        elif tag == "conv2d":
            if len(parts) < 8:
                raise ValueError(f"malformed conv2d layer in {filename!r}")
            in_ch, out_ch, kh, kw = (int(v) for v in parts[1:5])
            stride, padding, has_bias = (int(v) for v in parts[5:8])
            idx = _next_nonblank(lines, idx)
            counts = lines[idx].split()
            idx += 1
            if len(counts) < 2:
                raise ValueError(f"malformed conv2d counts in {filename!r}")
            weight_count, bias_count = int(counts[0]), int(counts[1])
            expected = out_ch * in_ch * kh * kw
            if weight_count != expected:
                raise ValueError(
                    f"conv2d weight count {weight_count} does not match "
                    f"{out_ch}x{in_ch}x{kh}x{kw} in {filename!r}"
                )
            if bias_count not in (0, out_ch):
                raise ValueError(
                    f"conv2d bias count {bias_count} is neither 0 nor {out_ch} "
                    f"in {filename!r}"
                )
            weights = [float(v) for v in lines[idx].split()]
            idx += 1
            if len(weights) != weight_count:
                raise ValueError(
                    f"conv2d weight line has {len(weights)} values, expected "
                    f"{weight_count} in {filename!r}"
                )
            biases = []
            if bias_count:
                biases = [float(v) for v in lines[idx].split()]
                idx += 1
                if len(biases) != bias_count:
                    raise ValueError(
                        f"conv2d bias line has {len(biases)} values, expected "
                        f"{bias_count} in {filename!r}"
                    )
            layers.append(
                Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=(kh, kw),
                    stride=(stride, stride),
                    padding=(padding, padding),
                    bias=bool(has_bias),
                    weights=weights,
                )
            )
            if biases:
                layers[-1]._bias = np.array(biases, dtype=float)
        elif tag in _STATELESS_LAYERS:
            factory, arg_names = _STATELESS_LAYERS[tag]
            if arg_names and arg_names[0] == "negative_slope":
                args = [float(v) for v in parts[1:]]
            else:
                args = [int(v) for v in parts[1:]]
            if len(args) != len(arg_names):
                raise ValueError(
                    f"{tag} layer expects {len(arg_names)} arguments, got "
                    f"{len(args)} in {filename!r}"
                )
            kwargs = dict(zip(arg_names, args))
            layers.append(factory(**kwargs) if kwargs else factory())
        else:
            raise ValueError(
                f"unknown layer tag {tag!r} in {filename!r}"
            )

    return Sequential(*layers)