"""Generic containers that chain or group other modules.

Both classes dispatch with ``hasattr`` and never name a concrete layer, so any
:class:`~core.nn.base.Module` can be nested in them.
"""

from .base import Module

__all__ = ["ModuleList", "Sequential"]


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out) if hasattr(layer, "__call__") else layer.forward(out)
        return out

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

    def named_modules(self):
        yield "", self
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "named_modules"):
                for name, mod in layer.named_modules():
                    if name:
                        yield f"{i}.{name}", mod
                    else:
                        yield f"{i}", mod
            else:
                yield str(i), layer

    def modules(self):
        yield self
        for _name, child in self.named_children():
            if hasattr(child, "modules"):
                yield from child.modules()
            else:
                yield child


class ModuleList(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = list(modules)

    def __iter__(self):
        return iter(self.modules_list)

    def __getitem__(self, idx):
        return self.modules_list[idx]

    def __len__(self):
        return len(self.modules_list)

    def append(self, module):
        self.modules_list.append(module)
        return self

    def named_children(self):
        for i, m in enumerate(self.modules_list):
            yield str(i), m

    def children(self):
        for _n, c in self.named_children():
            yield c

    def named_parameters(self):
        for i, m in enumerate(self.modules_list):
            if hasattr(m, "named_parameters"):
                for name, param in m.named_parameters():
                    if name:
                        yield f"{i}.{name}", param
                    else:
                        yield f"{i}", param
            elif hasattr(m, "parameters"):
                for j, param in enumerate(m.parameters()):
                    yield f"{i}.param{j}", param

    def parameters(self):
        for _n, p in self.named_parameters():
            yield p