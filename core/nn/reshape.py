"""Shape-only layers."""

import numpy as np

from .base import Module

__all__ = ["Flatten"]


class Flatten(Module):
    """Flatten to one dimension.

    Has no ``backward`` on purpose: its gradient is the identity reshape, so
    the caller's gradient flows through untouched. That makes "no backward" a
    deliberate signal here -- but not a reliable signal in general, since a
    missing ``backward`` elsewhere means a silently skipped derivative.
    """

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        return x.reshape(-1)