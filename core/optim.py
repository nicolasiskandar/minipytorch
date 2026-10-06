__all__ = ["SGD", "Adam"]

import numpy as np


def _param_key(p):
    """Stable identity for a trainable parameter.

    Shared modules (a layer listed twice in a Sequential) hand the optimizer
    several Parameter handles that wrap the same underlying storage. Keying on
    (layer, kind) instead of the handle object means a duplicate handle never
    double-applies gradients, and state survives the handles being rebuilt.
    """
    return id(p.layer), p.kind


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
        applied = set()
        for p in self.params:
            key = _param_key(p)
            if key in applied:
                continue
            applied.add(key)
            if self.momentum == 0.0:
                p.layer.apply_gradients(self.lr)
            else:
                p.apply_gradients(self.lr, self.momentum)
        return self


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = float(lr)
        beta1, beta2 = betas
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.params = list(params) if hasattr(params, "__iter__") else []
        for p in self.params:
            if not hasattr(p, "grad") or not hasattr(p, "value") or not hasattr(p, "set_value"):
                raise TypeError(
                    "Adam expects Parameter objects from Module.parameters(), "
                    f"got {type(p).__name__}"
                )
        self.m = {}
        self.v = {}
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
        return self

    def step(self):
        self.t += 1
        applied = set()
        for p in self.params:
            key = _param_key(p)
            if key in applied:
                continue
            applied.add(key)
            grad = np.asarray(p.grad(), dtype=float)
            if self.m.get(key) is None or self.m[key].shape != grad.shape:
                self.m[key] = np.zeros_like(grad)
            if self.v.get(key) is None or self.v[key].shape != grad.shape:
                self.v[key] = np.zeros_like(grad)

            g = grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * np.asarray(p.value, dtype=float)

            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)

            values = np.asarray(p.value, dtype=float)
            updated = values - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.set_value(updated)
        return self