"""Shape-only layers."""

import numpy as np

from .base import Module

__all__ = ["Flatten"]


class Flatten(Module):
    """Flatten a range of dimensions into a single axis, mirroring
    ``torch.nn.Flatten``.

    Defaults to ``start_dim=1``, ``end_dim=-1``: dim 0 is preserved and
    everything from dim 1 onward collapses into one axis. E.g. a
    ``(C, H, W)`` feature map collapses to ``(C, H*W)``.

    ``backward`` is the identity reshape: an incoming gradient with the same
    number of elements as the forward input is reshaped back to that input's
    shape, so a flatten sitting between (say) a Conv2d and a Linear hands
    gradients of the right shape to the pre-flatten layers.
    """

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self._last_input_shape = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        self._last_input_shape = x.shape
        start = self.start_dim
        end = self.end_dim
        if start < 0:
            start += x.ndim
        if end < 0:
            end += x.ndim
        if not (0 <= start <= end < x.ndim):
            raise ValueError(
                f"Flatten got start_dim={self.start_dim}, end_dim={self.end_dim} "
                f"for a {x.ndim}-D input"
            )
        out_shape = x.shape[:start] + (int(np.prod(x.shape[start : end + 1])),) + x.shape[end + 1 :]
        # Private copy, not a view: a user mutating the flattened output must
        # not write through into this layer's cached input.
        return x.reshape(out_shape).copy()

    def backward(self, dloss_dout):
        if self._last_input_shape is None:
            raise RuntimeError("Flatten.backward called before any forward")
        dloss_dout = np.asarray(dloss_dout, dtype=float)
        expected = int(np.prod(self._last_input_shape))
        if dloss_dout.size != expected:
            raise ValueError(
                f"Flatten.backward expected {expected} gradient values, "
                f"got {dloss_dout.size}"
            )
        return dloss_dout.reshape(self._last_input_shape)