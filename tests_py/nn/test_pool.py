"""MaxPool2d / AvgPool2d: values, strides, and numerical gradients.

The expected values below assume the flattened (C, H, W) C-order buffer, so a
flat index is *not* part of the first pooling window when C > 1.
"""

import numpy as np

from tests_py._helpers import EPS, TOL_WINDOW, central_diff


def test_maxpool2d_picks_window_maximum():
    from core import nn

    x = np.arange(16, dtype=float).reshape(1, 4, 4)
    out = nn.MaxPool2d(2)(x)
    assert out.shape == (1, 2, 2)
    assert np.allclose(out, np.array([[[5.0, 7.0], [13.0, 15.0]]]))


def test_avgpool2d_averages_window():
    from core import nn

    x = np.arange(16, dtype=float).reshape(1, 4, 4)
    out = nn.AvgPool2d(2)(x)
    assert out.shape == (1, 2, 2)
    assert np.allclose(out, np.array([[[2.5, 4.5], [10.5, 12.5]]]))


def test_pooling_strides():
    from core import nn

    x = np.zeros((1, 5, 5))
    assert nn.MaxPool2d(2)(x).shape == (1, 2, 2)
    assert nn.MaxPool2d(2, stride=1)(x).shape == (1, 4, 4)
    assert nn.AvgPool2d(3, stride=1)(x).shape == (1, 3, 3)


def test_pooling_rejects_input_smaller_than_kernel():
    from core import nn

    try:
        nn.MaxPool2d(5)(np.zeros((1, 3, 3)))
        raise AssertionError("expected ValueError for oversized kernel")
    except ValueError:
        pass


def test_pooling_gradients_numerical():
    from core import nn, losses

    np.random.seed(5)
    x = np.random.randn(2, 4, 4)
    y = np.random.randn(2, 2, 2)
    loss_fn = losses.MSELoss()

    for pool_cls in (nn.MaxPool2d, nn.AvgPool2d):
        pool = pool_cls(2)
        _loss, grad_out = loss_fn(pool(x), y)
        grad_in = pool.backward(grad_out)
        for pos in [(0, 0, 0), (1, 2, 3)]:
            x_plus = x.copy()
            x_plus[pos] += EPS
            x_minus = x.copy()
            x_minus[pos] -= EPS
            lp, _ = loss_fn(pool_cls(2)(x_plus), y)
            lm, _ = loss_fn(pool_cls(2)(x_minus), y)
            assert abs(central_diff(lp, lm) - grad_in[pos]) < TOL_WINDOW, (pool_cls, pos)