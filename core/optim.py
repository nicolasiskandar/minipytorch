__all__ = ["SGD"]


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.params = list(params) if hasattr(params, "__iter__") else []
        for p in self.params:
            if not hasattr(p, "apply_gradients"):
                raise TypeError(
                    "SGD expects Parameter objects from Module.parameters(), "
                    f"got {type(p).__name__}"
                )

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
        return self

    def step(self):
        if self.momentum == 0.0:
            applied = set()
            for p in self.params:
                key = id(p.layer)
                if key in applied:
                    continue
                applied.add(key)
                p.layer.apply_gradients(self.lr)
        else:
            for p in self.params:
                p.apply_gradients(self.lr, self.momentum)
        return self