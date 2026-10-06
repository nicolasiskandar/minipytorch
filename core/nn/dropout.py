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
        self._has_forward = False

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        self._has_forward = True
        self._last_input_shape = x.shape
        if not self.training() or self.p == 0.0:
            self._last_mask = None
            # Return a private copy so mutating the output (e.g. an in-place
            # activation upstream) cannot corrupt this layer's input buffer.
            return x.copy()
        keep_prob = 1.0 - self.p
        if keep_prob == 0.0:
            # p == 1: every unit is dropped. Record an all-zero mask so
            # backward multiplies by zero too, instead of passing the
            # gradient through untouched.
            self._last_mask = np.zeros_like(x)
            return self._last_mask.copy()
        mask = np.random.random(x.shape) < keep_prob
        self._last_mask = mask / keep_prob
        return x * self._last_mask

    def backward(self, dloss_dout):
        dloss_dout = np.asarray(dloss_dout, dtype=float)
        if self._last_mask is None:
            # Eval-mode (and p == 0) dropout is the identity, so the stored
            # mask is legitimately None and passing the gradient through is
            # correct. In training a missing mask means backward ran without a
            # forward, which would silently drop the whole model's gradient.
            if not self._has_forward and self.training() and self.p > 0:
                raise RuntimeError(
                    "Dropout.backward called before any forward while training"
                )
            return dloss_dout
        if dloss_dout.shape != self._last_mask.shape:
            if dloss_dout.size != self._last_mask.size:
                raise ValueError(
                    f"Dropout.backward expected {self._last_mask.size} "
                    f"gradient values matching {self._last_mask.shape}, "
                    f"got {dloss_dout.size} in shape {dloss_dout.shape}"
                )
            dloss_dout = dloss_dout.reshape(self._last_mask.shape)
        return dloss_dout * self._last_mask