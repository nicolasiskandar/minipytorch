"""Tag -> (factory, hyperparameter names) table for stateless layers.

``factory`` is always the layer class itself, never a wrapper: the writer finds
a layer's tag with ``type(layer) is factory``, so wrapping Softmax in a
zero-arg lambda made it unsaveable while the reader -- which calls
``factory()`` when there are no kwargs -- kept working.

``arg_names`` fixes the on-disk argument order, so reordering a tuple silently
changes what ``load_model`` expects.
"""

from ..nn import (
    AvgPool2d,
    Flatten,
    LeakyReLU,
    MaxPool2d,
    ReLU,
    ReLU6,
    Sigmoid,
    Softmax,
    Tanh,
)

_STATELESS_LAYERS = {
    "relu": (ReLU, ()),
    "sigmoid": (Sigmoid, ()),
    "tanh": (Tanh, ()),
    "relu6": (ReLU6, ()),
    "softmax": (Softmax, ()),
    "flatten": (Flatten, ("start_dim",)),
    "maxpool2d": (MaxPool2d, ("kernel_size", "stride", "padding")),
    "avgpool2d": (AvgPool2d, ("kernel_size", "stride", "padding")),
    "leaky_relu": (LeakyReLU, ("negative_slope",)),
}