"""Elementwise dropout with inverted scaling.

In training, each unit is kept with probability ``1 - p`` and survivors are
scaled by ``1 / (1 - p)``. That keeps the expected output equal to the input,
so a layer above sees an unchanged mean while individual units are thinned.

The keep/drop pattern is random and must be remembered for ``backward``: the
Jacobian of ``y = x * mask`` is ``mask``, so the incoming gradient is scaled by
the very same stored mask. Recomputing a fresh mask in backward would give an
unrelated gradient.

In eval the layer is the identity and stores no mask, so eval-mode forward is
deterministic -- which is the whole point of the train/eval distinction, and
why ``Module.train`` has to reach nested layers.
"""

import numpy as np

from .base import Module

__all__ = ["Dropout"]


class Dropout(Module):
    """Drop units at random while training; identity while evaluating."""

    def __init__(self, p=0.5):
        super().__init__()
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout p must be within [0, 1], got {p}")
        self.p = p
        self._last_mask = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if not self.training() or self.p == 0.0:
            self._last_mask = None
            return x
        keep_prob = 1.0 - self.p
        if keep_prob == 0.0:
            # p == 1: every unit is dropped. Record an all-zero mask so
            # backward multiplies by zero too, instead of passing the
            # gradient through untouched.
            self._last_mask = np.zeros_like(x)
            return self._last_mask
        mask = np.random.random(x.shape) < keep_prob
        self._last_mask = mask / keep_prob
        return x * self._last_mask

    def backward(self, dloss_dout):
        dloss_dout = np.asarray(dloss_dout, dtype=float)
        if self._last_mask is None:
            return dloss_dout
        return dloss_dout * self._last_mask