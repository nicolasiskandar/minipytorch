"""Loss functions: MSELoss and BCELoss.

CrossEntropyLoss is covered only by the export check in test_api_surface.py;
it has no behavioural test anywhere in the suite.
"""

import numpy as np


def test_mse_loss():
    from core import losses

    loss_fn = losses.MSELoss()
    loss, grad = loss_fn(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert abs(loss - 1.0) < 1e-10
    assert grad.shape == (2,)


def test_bce_loss():
    from core import losses

    loss_fn = losses.BCELoss()
    # simple case: pred 0.5, target 1.0 -> mean BCE is -log(0.5)
    loss, grad = loss_fn(np.array([0.5]), np.array([1.0]))
    assert abs(loss - (-np.log(0.5))) < 1e-10
    assert grad.shape == (1,)