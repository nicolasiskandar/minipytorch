import contextlib
import os
import tempfile

EPS = 1e-6

TOL_LINEAR = 1e-4
TOL_WINDOW = 1e-5


def central_diff(loss_plus, loss_minus, eps=EPS):
    """Central difference of the loss for one perturbed entry."""
    return (loss_plus - loss_minus) / (2.0 * eps)


@contextlib.contextmanager
def model_path(suffix=".mpk"):
    """Yield a path inside a temporary directory that is always cleaned up.

    Replaces the ``tempfile.mktemp()`` + try/finally + ``os.remove`` pattern
    that the serialization tests all repeated.
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "model" + suffix)


class FakeParam:
    """Minimal stand-in for ``nn.Parameter``, for optimizer formula tests.

    Adam is duck-typed against ``value`` / ``grad()`` / ``set_value``, so a real
    Parameter is not needed to pin the update rule.
    """

    kind = "weight"
    layer = None

    def __init__(self, value, grad):
        import numpy as np

        self._value = np.array([value], dtype=float)
        self._grad = np.array([grad], dtype=float)

    @property
    def value(self):
        return self._value

    def grad(self):
        return self._grad

    def set_value(self, values):
        import numpy as np

        self._value = np.asarray(values, dtype=float)
        return self

    def zero_grad(self):
        import numpy as np

        self._grad = np.zeros_like(self._grad)

    def apply_gradients(self, lr, momentum):
        return self