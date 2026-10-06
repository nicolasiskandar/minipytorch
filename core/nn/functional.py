"""Stateless functional forms of the layers.

Each function is the arithmetic behind a module, with no stored state and no
``backward``. The caller owns the gradient, so these suit one-off use inside a
custom layer or an experiment; use the module from :mod:`core.nn.dropout` when
the gradient has to flow through a trained network, because the module is what
remembers the keep/drop mask that ``backward`` needs.
"""

import numpy as np

__all__ = ["dropout"]


def dropout(x, p=0.5, training=True):
    """Drop units at random while ``training``, identity otherwise.

    Mirrors :class:`core.nn.dropout.Dropout` forward, including the ``p``
    range check and the inverted ``1 / (1 - p)`` scaling, so the two paths
    cannot silently disagree on what an out-of-range ``p`` means.
    """
    x = np.asarray(x, dtype=float)
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout p must be within [0, 1], got {p}")
    if not training or p == 0.0:
        return x
    keep_prob = 1.0 - p
    if keep_prob == 0.0:
        return np.zeros_like(x)
    mask = np.random.random(x.shape) < keep_prob
    return x * (mask / keep_prob)