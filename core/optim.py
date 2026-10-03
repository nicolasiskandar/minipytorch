__all__ = ["SGD"]


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.lr = float(lr)
        self.momentum = float(momentum)
        # params can be iterable of modules or parameter containers
        try:
            self.params = list(params)
        except TypeError:
            self.params = list(params) if hasattr(params, "__iter__") else []

    def zero_grad(self):
        # No-op for current design (grads overwritten on backward)
        return self

    def step(self):
        for p in self.params:
            if hasattr(p, "apply_gradients"):
                p.apply_gradients(self.lr)
            # if it's a tensor-like, no op
        return self
