__all__ = ["save_model", "load_model"]

import os
from typing import Any

import numpy as np

from .nn import Linear, Sequential
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


def _get_activation_name(kind: ActivationKind) -> str:
    name = _ACTIVATION_NAMES.get(int(kind))
    if name is None:
        raise ValueError(f"no save format for activation kind {int(kind)}")
    return name


def save_model(model: Any, filename: str) -> None:
    if isinstance(model, Sequential):
        layers = model.layers
    else:
        # treat as single layer module
        layers = [model]

    with open(filename, "w", encoding="utf-8") as out:
        out.write(f"{len(layers)}\n")
        for layer in layers:
            if not isinstance(layer, Linear):
                raise TypeError("save_model only supports Sequential of Linear layers")
            w = np.array(layer.weights(), dtype=float).reshape(layer.out_features, layer.in_features)
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


def load_model(filename: str) -> Sequential:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0
    # number of layers
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    num_layers = int(lines[idx].strip())
    idx += 1

    layers = []
    for _ in range(num_layers):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        num_neurons = int(lines[idx].strip())
        idx += 1
        weights_list = []
        biases_list = []
        act_kind = ActivationKind.SIGMOID
        in_features = 0
        for i in range(num_neurons):
            line = lines[idx].rstrip()
            idx += 1
            # split by spaces
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
            # bias is next
            bias_val = float(parts[1])
            # weights
            wparts = parts[2:] if len(parts) > 2 else []
            weights = [float(p) for p in wparts]
            if in_features == 0 and len(weights) > 0:
                in_features = len(weights)
            weights_list.extend(weights)
            biases_list.append(bias_val)
        # build linear: out_features = num_neurons, in_features known
        out_features = num_neurons
        if in_features == 0:
            in_features = 1
        layer = Linear(
            in_features=in_features,
            out_features=out_features,
            activation=act_kind,
            weights=weights_list,
            bias=biases_list,
        )
        layers.append(layer)

    return Sequential(*layers)
