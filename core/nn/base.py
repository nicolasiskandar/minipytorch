"""Base class shared by every layer in :mod:`core.nn`."""

__all__ = ["Module"]


class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        for _name, _param in self.named_parameters():
            yield _param

    def named_parameters(self):
        return []

    def children(self):
        for _name, _child in self.named_children():
            yield _child

    def named_children(self):
        return []

    def modules(self):
        yield self
        for child in self.children():
            if hasattr(child, "modules"):
                yield from child.modules()
            else:
                yield child

    def named_modules(self):
        yield "", self
        for name, child in self.named_children():
            if hasattr(child, "named_modules"):
                for cname, cmod in child.named_modules():
                    if cname:
                        yield f"{name}.{cname}", cmod
                    else:
                        yield name, cmod
            else:
                yield name, child

    def zero_grad(self):
        return self

    def train(self, mode=True):
        self._training = mode
        return self

    def eval(self):
        self._training = False
        return self

    def training(self):
        return getattr(self, "_training", True)