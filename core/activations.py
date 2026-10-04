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
    leaky_relu_f64,
    leaky_relu_derivative_from_output_f64,
)

__all__ = [
    "ActivationKind",
    "sigmoid",
    "tanh",
    "relu",
    "sigmoid_deriv",
    "tanh_deriv",
    "relu_deriv",
    "relu6",
    "relu6_deriv",
    "leaky_relu",
    "leaky_relu_deriv",
    "softmax",
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
    except (TypeError, ValueError):
        import numpy as np

        return np.vectorize(relu6_f64)(np.array(x, dtype=float))


def relu6_deriv(y):
    try:
        return relu6_deriv_from_output_f64(float(y))
    except (TypeError, ValueError):
        import numpy as np

        return np.vectorize(relu6_deriv_from_output_f64)(np.array(y, dtype=float))


def leaky_relu(x, alpha=0.01):
    try:
        return leaky_relu_f64(float(x))
    except (TypeError, ValueError):
        import numpy as np

        x = np.array(x, dtype=float)
        if alpha != 0.01:
            return np.where(x > 0.0, x, alpha * x)
        return np.vectorize(leaky_relu_f64)(x)


def leaky_relu_deriv(y, alpha=0.01):
    try:
        return leaky_relu_derivative_from_output_f64(float(y))
    except (TypeError, ValueError):
        import numpy as np

        y = np.array(y, dtype=float)
        if alpha != 0.01:
            return np.where(y > 0.0, 1.0, alpha)
        return np.vectorize(leaky_relu_derivative_from_output_f64)(y)


def softmax(x, dim=-1):
    import numpy as np

    x = np.array(x, dtype=float)
    if x.ndim == 0:
        return np.array(1.0, dtype=float)
    dim = dim % x.ndim
    shifted = x - np.max(x, axis=dim, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=dim, keepdims=True)
