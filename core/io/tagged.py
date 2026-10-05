"""The tagged ``minipytorch-v2`` format: per-layer type plus hyperparameters.

Reached for anything that is not a Linear-only network. Every reader path
validates its counts against the declared shape instead of trusting the file,
and every float argument is parsed as a float -- routing one through ``int()``
once truncated ``LeakyReLU.negative_slope=0.25`` to ``0``.
"""

import numpy as np

from ..nn import Conv2d, Linear, Sequential
from .format import (
    _FORMAT_TAG,
    _fmt,
    _get_activation_name,
    _next_nonblank,
    _REVERSE_ACTIVATION_NAMES,
)
from .registry import _STATELESS_LAYERS


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