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

    def zero_grad(self):
        """Clear gradients here and in every child.

        Recurses rather than returning early: a bare ``return self`` leaves a
        container's own (empty) gradients cleared while its children keep
        theirs, so ``net.zero_grad()`` silently did nothing.
        """
        for child in self.children():
            child.zero_grad()
        return self

    def train(self, mode=True):
        """Set this module's mode and every child's, returning self.

        Recursion is required, not cosmetic: only layers that read
        ``training()`` care, but ``Sequential.eval()`` that leaves a nested
        layer in training mode makes eval-mode inference stochastic.
        """
        self._training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def training(self):
        return getattr(self, "_training", True)