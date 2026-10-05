"""Layer definitions and the :class:`Module` base.

This module is the aggregate: import it (``from core import nn``) rather than
reaching into the leaves. Everything listed in ``__all__`` is re-exported here
by name, and ``core/__init__.py`` star-imports this module -- so a submodule
name left out of ``__all__`` on purpose, otherwise it would shadow a real
``core`` module. ``core.nn.activations`` is the prime example: binding it into
``core`` would replace the functional kernels in ``core.activations``.

Deliberately *not* in ``__all__`` but kept bound, because existing code reaches
for them:

* ``ActivationKind`` -- used as ``nn.ActivationKind.RELU6``.
* ``FastLayer`` and the ``conv2d_*`` / ``*pool2d_*`` native symbols -- the
  native extension's surface, re-exported unchanged rather than re-wrapped.
* ``_ElementwiseActivation`` -- the shared base whose ``backward`` gates the
  gradient chain of every activation module.
"""

# Leaf modules, bottom-up. `activations` last: it reaches back into the
# sibling package `core.activations` for its kernels.
from .base import Module
from .params import Parameter, _ArrayParameter
from ._shape import _pair
from .linear import Linear
from .containers import ModuleList, Sequential
from .reshape import Flatten
from .conv import Conv2d
from .pool import AvgPool2d, MaxPool2d
from .activations import (
    LeakyReLU,
    ReLU,
    ReLU6,
    Sigmoid,
    Softmax,
    Tanh,
    _ElementwiseActivation,
)

from .._minipytorch import ActivationKind, FastLayer
from .._minipytorch import (
    avgpool2d_backward,
    avgpool2d_forward,
    conv2d_backward_input,
    conv2d_backward_weight,
    conv2d_forward,
    maxpool2d_backward,
    maxpool2d_forward,
)

__all__ = [
    "Module",
    "Parameter",
    "ModuleList",
    "Linear",
    "Sequential",
    "Softmax",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "ReLU6",
    "Flatten",
    "Conv2d",
    "MaxPool2d",
    "AvgPool2d",
]