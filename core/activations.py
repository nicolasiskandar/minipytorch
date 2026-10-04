from enum import Enum

from ._minipytorch import (
    ActivationKind,
    sigmoid_f64,
    tanh_f64,
    relu_f64,
    sigmoid_deriv_from_output_f64,
    tanh_deriv_from_output_f64,
    relu_deriv_from_output_f64,
    relu6_f64,
    relu6_deriv_from_output_f64,
)

__all__ = [
    "ActivationKind",
    "sigmoid",
    "tanh",
    "relu",
    "sigmoid_deriv",
    "tanh_deriv",
    "relu_deriv",
]


def sigmoid(x):
    try:
        return sigmoid_f64(float(x))
    except Exception:
        import numpy as np

        return np.vectorize(sigmoid_f64)(np.array(x, dtype=float))


def tanh(x):
    try:
        return tanh_f64(float(x))
    except Exception:
        import numpy as np

        return np.vectorize(tanh_f64)(np.array(x, dtype=float))


def relu(x):
    try:
        return relu_f64(float(x))
    except Exception:
        import numpy as np

        return np.vectorize(relu_f64)(np.array(x, dtype=float))


def sigmoid_deriv(y):
    try:
        return sigmoid_deriv_from_output_f64(float(y))
    except Exception:
        import numpy as np

        return np.vectorize(sigmoid_deriv_from_output_f64)(np.array(y, dtype=float))


def tanh_deriv(y):
    try:
        return tanh_deriv_from_output_f64(float(y))
    except Exception:
        import numpy as np

        return np.vectorize(tanh_deriv_from_output_f64)(np.array(y, dtype=float))


def relu_deriv(y):
    try:
        return relu_deriv_from_output_f64(float(y))
    except Exception:
        import numpy as np

        return np.vectorize(relu_deriv_from_output_f64)(np.array(y, dtype=float))

def relu6(x):
    try:
        return relu6_f64(float(x))
    except Exception:
        import numpy as np
        return np.vectorize(relu6_f64)(np.array(x, dtype=float))


def relu6_deriv(y):
    try:
        return relu6_deriv_from_output_f64(float(y))
    except Exception:
        import numpy as np
        return np.vectorize(relu6_deriv_from_output_f64)(np.array(y, dtype=float))


def softmax(x, dim=None):
    import numpy as np
    x = np.array(x, dtype=float)
    if x.ndim == 0:
        return np.array(1.0, dtype=float)
    if dim is None:
        # default to -1
        x_shift = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x_shift)
        return e / np.sum(e, axis=-1, keepdims=True)
    else:
        x_shift = x - np.max(x, axis=dim, keepdims=True)
        e = np.exp(x_shift)
        return e / np.sum(e, axis=dim, keepdims=True)
