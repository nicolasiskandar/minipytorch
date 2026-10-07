"""The generic container that chains other modules.

Forward dispatch is with ``hasattr`` and never names a concrete layer, so any
:class:`~core.nn.base.Module` can be nested in it. Backward is strict: a child
without a ``backward()`` raises instead of being silently skipped, which would
otherwise pass the gradient straight through dead units.
"""

from .base import Module

__all__ = ["Sequential"]


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out) if hasattr(layer, "__call__") else layer.forward(out)
        return out

    def backward(self, dloss_dout):
        grad = dloss_dout
        for i, layer in reversed(list(enumerate(self.layers))):
            if not callable(getattr(layer, "backward", None)):
                raise RuntimeError(
                    f"Sequential.backward: layer {i} "
                    f"({type(layer).__name__}) has no backward()"
                )
            grad = layer.backward(grad)
        return grad

    def named_children(self):
        for i, layer in enumerate(self.layers):
            yield str(i), layer

    def children(self):
        for _name, child in self.named_children():
            yield child

    def named_parameters(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "named_parameters"):
                for name, param in layer.named_parameters():
                    if name:
                        yield f"{i}.{name}", param
                    else:
                        yield f"{i}", param
            elif hasattr(layer, "parameters"):
                for j, param in enumerate(layer.parameters()):
                    yield f"{i}.param{j}", param

    def parameters(self):
        for _name, _param in self.named_parameters():
            yield _param

    def modules(self):
        yield self
        for _name, child in self.named_children():
            if hasattr(child, "modules"):
                yield from child.modules()
            else:
                yield child
