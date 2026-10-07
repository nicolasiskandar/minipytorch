"""Conv2d: shapes, hand-computed values, parameters, and numerical gradients."""

import numpy as np

from tests_py._helpers import EPS, TOL_WINDOW, central_diff


def test_conv2d_forward_shape_and_hand_computed_value():
    from core import nn

    conv = nn.Conv2d(1, 1, 2, weights=[1.0, 1.0, 1.0, 1.0])
    conv._bias = np.zeros(1)
    out = conv(np.arange(9, dtype=float).reshape(1, 3, 3))
    assert out.shape == (1, 2, 2)
    assert np.allclose(out, np.array([[[8.0, 12.0], [20.0, 24.0]]]))


def test_conv2d_stride_and_padding_shapes():
    from core import nn

    x = np.zeros((1, 5, 5))
    assert nn.Conv2d(1, 1, 3)(x).shape == (1, 3, 3)
    assert nn.Conv2d(1, 1, 3, stride=2)(x).shape == (1, 2, 2)
    assert nn.Conv2d(1, 1, 3, padding=1)(x).shape == (1, 5, 5)


def test_conv2d_parameters_expose_weight_and_bias():
    from core import nn, losses, optim

    conv = nn.Conv2d(1, 2, 3)
    named = dict((n, p) for n, p in conv.named_parameters())
    assert set(named) == {"weight", "bias"}
    assert named["weight"].value.shape == (2, 1, 3, 3)
    assert named["bias"].value.shape == (2,)

    out = conv(np.ones((1, 5, 5)))
    _loss, g = losses.MSELoss()(out, np.zeros_like(out))
    conv.backward(g)
    before = conv._weights.copy()
    optim.SGD(list(conv.parameters()), lr=0.1).step()
    assert not np.allclose(conv._weights, before)


def test_conv2d_rejects_wrong_rank_input():
    from core import nn

    try:
        nn.Conv2d(1, 1, 3)(np.zeros(9))
        raise AssertionError("expected ValueError for 1D input")
    except ValueError:
        pass
    try:
        nn.Conv2d(2, 1, 3)(np.zeros((1, 4, 4)))
        raise AssertionError("expected ValueError for channel mismatch")
    except ValueError:
        pass


def test_conv2d_gradient_numerical():
    from core import nn, losses

    np.random.seed(3)
    conv = nn.Conv2d(2, 2, 3, padding=1)
    x = np.random.randn(2, 4, 4)
    y = np.random.randn(2, 4, 4)
    loss_fn = losses.MSELoss()

    _loss, grad_out = loss_fn(conv(x), y)
    grad_in = conv.backward(grad_out)

    w = conv._weights.copy()
    for idx in [(0, 0, 0, 0), (1, 1, 2, 2)]:
        w_plus = w.copy()
        w_plus[idx] += EPS
        w_minus = w.copy()
        w_minus[idx] -= EPS
        plus = nn.Conv2d(2, 2, 3, padding=1, weights=w_plus.reshape(-1))
        plus._bias = conv._bias.copy()
        minus = nn.Conv2d(2, 2, 3, padding=1, weights=w_minus.reshape(-1))
        minus._bias = conv._bias.copy()
        lp, _ = loss_fn(plus(x), y)
        lm, _ = loss_fn(minus(x), y)
        assert abs(central_diff(lp, lm) - conv._grad_weights[idx]) < TOL_WINDOW, idx

    x_plus = x.copy()
    x_plus[0, 1, 1] += EPS
    x_minus = x.copy()
    x_minus[0, 1, 1] -= EPS
    lp, _ = loss_fn(conv(x_plus), y)
    lm, _ = loss_fn(conv(x_minus), y)
    assert abs(central_diff(lp, lm) - grad_in[0, 1, 1]) < TOL_WINDOW

def test_default_convs_do_not_share_weights():
    """Same-shaped default Conv2ds used to start identical (fresh rng(0) each)."""
    from core import nn

    def weight_of(conv):
        return np.asarray(dict(conv.named_parameters())["weight"].value)

    first = nn.Conv2d(1, 2, 3)
    second = nn.Conv2d(1, 2, 3)
    assert not np.allclose(weight_of(first), weight_of(second))
