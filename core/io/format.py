"""On-disk format constants and small text helpers."""

from ..activations import ActivationKind

_FORMAT_TAG = "minipytorch-v2"

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


def _fmt(values):
    """%.17g, so a float survives a write/read cycle bit-for-bit."""
    return " ".join(f"{float(v):.17g}" for v in values)


def _next_nonblank(lines, idx):
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return idx