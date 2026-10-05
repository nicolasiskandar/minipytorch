"""2D convolution over a single (unbatched) C x H x W input."""

import numpy as np

from .._minipytorch import (
    conv2d_backward_input,
    conv2d_backward_weight,
    conv2d_forward,
)
from ._shape import _pair
from .base import Module
from .params import _ArrayParameter

__all__ = ["Conv2d"]


class Conv2d(Module):
    """2D convolution over a single (unbatched) C x H x W input.

    Mirrors torch.nn.Conv2d's argument names but operates on one sample at a
    time, like Linear. Weights are stored as
    (out_channels, in_channels, kernel_height, kernel_width).

    ``padding`` pads both spatial dims of the *output*, height included. A 1x3
    input with a 1x1 kernel and ``padding=1`` yields 3x5, not 1x3, and the real
    values land on output row 1. Derive expected indices from the returned
    shape, never from "same padding" intuition.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        weights=None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.use_bias = bool(bias)

        kh, kw = self.kernel_size
        shape = (self.out_channels, self.in_channels, kh, kw)
        if weights is None:
            rng = np.random.default_rng(0)
            w = rng.standard_normal(shape) * (1.0 / np.sqrt(self.in_channels * kh * kw))
        else:
            w = np.array(weights, dtype=float).reshape(shape)
        self._weights = w
        self._grad_weights = np.zeros(shape)
        self._bias = np.zeros(self.out_channels) if self.use_bias else np.zeros(0)
        self._grad_bias = np.zeros(self.out_channels) if self.use_bias else np.zeros(0)
        self._last_input = None
        self._last_input_shape = None

    def named_parameters(self):
        yield "weight", _ArrayParameter(self, "weight")
        if self.use_bias:
            yield "bias", _ArrayParameter(self, "bias")

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError(
                f"Conv2d expects a 3D (C, H, W) input, got shape {x.shape}"
            )
        c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"Conv2d expected {self.in_channels} channels, got {c}"
            )
        kh, kw = self.kernel_size
        if h + 2 * self.padding[0] < kh or w + 2 * self.padding[1] < kw:
            raise ValueError(
                f"Conv2d input {h}x{w} is smaller than kernel {kh}x{kw}"
            )
        self._last_input = x.reshape(-1).copy()
        self._last_input_shape = (c, h, w)
        flat_out, out_shape = conv2d_forward(
            self._last_input,
            self._weights.reshape(-1),
            self._bias,
            c,
            h,
            w,
            self.out_channels,
            kh,
            kw,
            self.stride[0],
            self.padding[0],
        )
        return flat_out.reshape(out_shape)

    def backward(self, dloss_dout):
        dloss_dout = np.asarray(dloss_dout, dtype=float).reshape(-1)
        if self._last_input is None:
            raise RuntimeError("Conv2d.backward called before any forward")
        c, h, w = self._last_input_shape
        kh, kw = self.kernel_size
        grad_w, grad_b = conv2d_backward_weight(
            dloss_dout,
            self._last_input,
            c,
            h,
            w,
            self.out_channels,
            kh,
            kw,
            self.stride[0],
            self.padding[0],
        )
        self._grad_weights = grad_w.reshape(self._weights.shape)
        self._grad_bias = grad_b
        grad_in = conv2d_backward_input(
            dloss_dout,
            self._weights.reshape(-1),
            c,
            h,
            w,
            self.out_channels,
            kh,
            kw,
            self.stride[0],
            self.padding[0],
        )
        return grad_in.reshape(self._last_input_shape)

    def zero_grad(self):
        self._grad_weights = np.zeros_like(self._weights)
        self._grad_bias = np.zeros_like(self._bias)
        return self

    def apply_gradients(self, lr=0.01):
        self._weights = self._weights - lr * self._grad_weights
        if self.use_bias:
            self._bias = self._bias - lr * self._grad_bias
        return self