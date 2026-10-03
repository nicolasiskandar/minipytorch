import numpy as np

from ._minipytorch import mse_loss

__all__ = ["MSELoss", "mse_loss"]


class MSELoss:
    def forward(self, pred, target):
        loss, grad_pred = mse_loss(
            np.array(pred, dtype=float), np.array(target, dtype=float)
        )
        return float(loss), grad_pred

    def __call__(self, pred, target):
        return self.forward(pred, target)
