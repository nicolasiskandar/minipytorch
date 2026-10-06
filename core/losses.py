import numpy as np

from ._minipytorch import mse_loss, bce_loss

__all__ = ["MSELoss", "BCELoss", "CrossEntropyLoss", "mse_loss", "bce_loss"]


class MSELoss:
    def forward(self, pred, target):
        pred = np.array(pred, dtype=float)
        target = np.array(target, dtype=float)
        n = pred.size
        if n == 0:
            return 0.0, pred
        # The native kernel returns 0.5 * sum((p - t)^2) with grad (p - t).
        # Scale to PyTorch's 'mean' reduction so the loss magnitude does not
        # grow with the batch size: mean((p - t)^2) and 2*(p - t)/n.
        loss, grad = mse_loss(pred, target)
        scale = 2.0 / n
        return float(loss * scale), grad * scale

    def __call__(self, pred, target):
        return self.forward(pred, target)


class BCELoss:
    def forward(self, pred, target):
        loss, grad_pred = bce_loss(
            np.array(pred, dtype=float), np.array(target, dtype=float)
        )
        return float(loss), grad_pred

    def __call__(self, pred, target):
        return self.forward(pred, target)


class CrossEntropyLoss:
    def __init__(self, ignore_index=-100):
        self.ignore_index = ignore_index

    def forward(self, pred, target, eps=1e-12):
        pred = np.array(pred, dtype=float)
        target = np.array(target, dtype=float)
        if pred.size == 0:
            return 0.0, pred
        if pred.ndim == 2:
            N, C = pred.shape
            p = pred - np.max(pred, axis=1, keepdims=True)
            p = np.exp(p)
            p = p / (np.sum(p, axis=1, keepdims=True) + 1e-300)
            loss = 0.0
            grad = np.zeros_like(pred)
            kept = 0
            for i in range(N):
                t = target.flatten()[i]
                if t == self.ignore_index:
                    continue
                kept += 1
                c = int(round(float(t)))
                if c < 0 or c >= C:
                    c = 0
                pc = p[i, c]
                loss -= np.log(pc + eps)
                grad[i] = p[i].copy()
                grad[i, c] -= 1.0
            if kept:
                loss /= kept
                grad /= kept
            return float(loss), grad
        else:
            p = pred - np.max(pred)
            p = np.exp(p)
            p = p / (np.sum(p) + 1e-300)
            t = target.flatten()[0] if target.size > 0 else 0
            c = int(round(float(t)))
            if c < 0 or c >= p.shape[0]:
                c = 0
            loss = -np.log(p[c] + eps)
            grad = p.copy()
            grad[c] -= 1.0
            return float(loss), grad

    def __call__(self, pred, target, *args, **kwargs):
        return self.forward(pred, target)
