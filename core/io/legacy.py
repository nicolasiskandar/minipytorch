"""The original positional format, for Linear-only networks.

Kept byte-for-byte: the text is load-bearing, several tests assert on it
verbatim, and old checkpoints written by earlier versions have to still load.
Anything that is not exclusively :class:`~core.nn.Linear` is rejected here
rather than being coerced.
"""

import numpy as np

from ..activations import ActivationKind
from ..nn import Linear, Sequential
from .format import _get_activation_name, _next_nonblank, _REVERSE_ACTIVATION_NAMES


def _is_legacy_linear_only(layers):
    return all(isinstance(layer, Linear) for layer in layers)


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