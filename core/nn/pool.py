"""2D max and average pooling over a single (unbatched) C x H x W input.

Gradients are only as good as the shape bookkeeping: the flattened buffer is
``(C, H, W)`` C-order, so flat index 3 is ``(0, 0, 3)`` and is *not* part of the
first pooling window.
"""

import numpy as np

from .._minipytorch import (
    avgpool2d_backward,
    avgpool2d_forward,
    maxpool2d_backward,
    maxpool2d_forward,
)
from ._shape import _pair
from .base import Module

__all__ = ["AvgPool2d", "MaxPool2d"]


class MaxPool2d(Module):
    """2D max pooling over a single (unbatched) C x H x W input."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            self.stride = _pair(kernel_size)
        else:
            self.stride = _pair(stride)
        self.padding = _pair(padding)
        self._last_shape = None
        self._last_out_shape = None
        self._indices = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError(
                f"MaxPool2d expects a 3D (C, H, W) input, got shape {x.shape}"
            )
        c, h, w = x.shape
        if h + 2 * self.padding[0] < self.kernel_size[0] or w + 2 * self.padding[1] < self.kernel_size[1]:
            raise ValueError(
                f"MaxPool2d input {h}x{w} is smaller than kernel {self.kernel_size}"
            )
        out, indices, out_shape = maxpool2d_forward(
            x.reshape(-1),
            c,
            h,
            w,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
        )
        self._last_shape = (c, h, w)
        self._last_out_shape = tuple(int(v) for v in out_shape)
        self._indices = indices
        return out.reshape(self._last_out_shape)

    def backward(self, dloss_dout):
        dloss_dout = np.asarray(dloss_dout, dtype=float).reshape(-1)
        if self._indices is None:
            raise RuntimeError("MaxPool2d.backward called before any forward")
        c, h, w = self._last_shape
        oc, oh, ow = self._last_out_shape
        grad_in = maxpool2d_backward(
            dloss_dout, self._indices, c, h, w, oc, oh, ow
        )
        return grad_in.reshape(c, h, w)


class AvgPool2d(Module):
    """2D average pooling over a single (unbatched) C x H x W input."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        if stride is None:
            self.stride = _pair(kernel_size)
        else:
            self.stride = _pair(stride)
        self.padding = _pair(padding)
        self._last_shape = None
        self._last_out_shape = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError(
                f"AvgPool2d expects a 3D (C, H, W) input, got shape {x.shape}"
            )
        c, h, w = x.shape
        if h + 2 * self.padding[0] < self.kernel_size[0] or w + 2 * self.padding[1] < self.kernel_size[1]:
            raise ValueError(
                f"AvgPool2d input {h}x{w} is smaller than kernel {self.kernel_size}"
            )
        out, out_shape = avgpool2d_forward(
            x.reshape(-1),
            c,
            h,
            w,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
        )
        self._last_shape = (c, h, w)
        self._last_out_shape = tuple(int(v) for v in out_shape)
        return out.reshape(self._last_out_shape)

    def backward(self, dloss_dout):
        dloss_dout = np.asarray(dloss_dout, dtype=float).reshape(-1)
        if self._last_shape is None:
            raise RuntimeError("AvgPool2d.backward called before any forward")
        c, h, w = self._last_shape
        oc, oh, ow = self._last_out_shape
        grad_in = avgpool2d_backward(
            dloss_dout,
            c,
            h,
            w,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
            oc,
            oh,
            ow,
        )
        return grad_in.reshape(c, h, w)