"""Linear layer: forward, backward shapes, and numerical gradients."""

import numpy as np

from tests_py._helpers import EPS, TOL_LINEAR, central_diff


def test_linear_forward():
    from core import nn

    l = nn.Linear(2, 3)
    out = l.forward(np.array([1.0, 2.0]))
    assert out.shape == (3,)


def test_backward_shapes():
    from core import nn, losses

    l1 = nn.Linear(2, 3, activation='relu')
    l2 = nn.Linear(3, 1, activation='sigmoid')
    h = l1(np.array([1.0, -0.5]))
    out = l2(h)
    loss_fn = losses.MSELoss()
    loss, g = loss_fn(out, np.array([1.0]))
    gh = l2.backward(g)
    gxh = l1.backward(gh)
    assert gxh.shape == (2,)


def test_weights_roundtrip():
    from core import nn

    l = nn.Linear(2, 3, weights=[1, 2, 3, 4, 5, 6], bias=[0.1, 0.2, 0.3])
    w = np.array(l.weights())
    assert w.shape == (6,)
    assert abs(w[0] - 1.0) < 1e-12
    l.set_weights([0, 1, 2, 3, 4, 5])
    w2 = np.array(l.weights())
    assert abs(w2[0] - 0.0) < 1e-12


def test_linear_activation_parity_shapes():
    from core import nn

    x = np.array([1.0, -0.5])
    for act in ['sigmoid', 'tanh', 'relu']:
        l = nn.Linear(2, 3, activation=act)
        out = l(x)
        assert out.shape == (3,)


def test_numerical_gradients_mse():
    from core import nn, losses

    np.random.seed(123)
    l = nn.Linear(2, 2, activation='tanh', weights=np.array([0.1, -0.2, 0.3, 0.4]), bias=np.array([0.0, 0.0]))
    x = np.array([1.0, 0.5])
    y = np.array([0.2, 0.8])
    # forward/backward
    h = l(x)
    loss_fn = losses.MSELoss()
    loss, g = loss_fn(h, y)
    gh = l.backward(g)
    # numerical grad for weight 0
    w = np.array(l.weights(), dtype=float)
    # perturb w[0]
    w1 = w.copy(); w1[0] += EPS
    l2 = nn.Linear(2, 2, activation='tanh', weights=w1, bias=np.array([0.0, 0.0]))
    loss1, _ = loss_fn(l2(x), y)
    w2 = w.copy(); w2[0] -= EPS
    l3 = nn.Linear(2, 2, activation='tanh', weights=w2, bias=np.array([0.0, 0.0]))
    loss2, _ = loss_fn(l3(x), y)
    num_grad = central_diff(loss1, loss2)
    anal = np.array(l.grad_weights())[0]
    assert abs(num_grad - anal) < TOL_LINEAR


def test_numerical_gradients_bce():
    from core import nn, losses

    np.random.seed(42)
    l = nn.Linear(2, 1, activation='sigmoid', weights=np.array([0.2, -0.3]), bias=np.array([0.1]))
    x = np.array([0.7, 0.4])
    y = np.array([1.0])
    loss_fn = losses.BCELoss()
    h = l(x)
    loss, g = loss_fn(h, y)
    _ = l.backward(g)
    w = np.array(l.weights(), dtype=float)
    # w[0]
    w1 = w.copy(); w1[0] += EPS
    l2 = nn.Linear(2, 1, activation='sigmoid', weights=w1, bias=np.array([0.1]))
    loss1, _ = loss_fn(l2(x), y)
    w2 = w.copy(); w2[0] -= EPS
    l3 = nn.Linear(2, 1, activation='sigmoid', weights=w2, bias=np.array([0.1]))
    loss2, _ = loss_fn(l3(x), y)
    num_grad = central_diff(loss1, loss2)
    anal = np.array(l.grad_weights())[0]
    assert abs(num_grad - anal) < TOL_LINEAR


def test_linear_grad_biases_numerical():
    from core import nn, losses

    np.random.seed(7)
    l = nn.Linear(2, 2, activation='tanh', weights=np.array([0.1, -0.2, 0.3, 0.4]), bias=np.array([0.05, -0.1]))
    x = np.array([0.8, -0.3])
    y = np.array([0.4, 0.6])
    loss_fn = losses.MSELoss()
    h = l(x)
    loss, g = loss_fn(h, y)
    _ = l.backward(g)
    b = np.array(l.biases(), dtype=float)
    # bias[0]
    b1 = b.copy(); b1[0] += EPS
    l2 = nn.Linear(2, 2, activation='tanh', weights=l.weights(), bias=b1)
    loss1, _ = loss_fn(l2(x), y)
    b2 = b.copy(); b2[0] -= EPS
    l3 = nn.Linear(2, 2, activation='tanh', weights=l.weights(), bias=b2)
    loss2, _ = loss_fn(l3(x), y)
    num_grad = central_diff(loss1, loss2)
    anal = np.array(l.grad_biases())[0]
    assert abs(num_grad - anal) < TOL_LINEAR


def test_leaky_relu_gradient_numerical():
    from core import nn, losses

    w = np.array([0.2, -0.3, 0.1, 0.4])
    b = np.array([0.05, -0.1])
    x = np.array([0.8, -0.3])
    y = np.array([0.4, 0.6])
    loss_fn = losses.MSELoss()

    layer = nn.Linear(2, 2, activation='leaky_relu', weights=w, bias=b)
    out = layer(x)
    _loss, grad_out = loss_fn(out, y)
    _ = layer.backward(grad_out)

    w_plus = w.copy()
    w_plus[0] += EPS
    w_minus = w.copy()
    w_minus[0] -= EPS
    loss_plus, _ = loss_fn(nn.Linear(2, 2, activation='leaky_relu', weights=w_plus, bias=b)(x), y)
    loss_minus, _ = loss_fn(nn.Linear(2, 2, activation='leaky_relu', weights=w_minus, bias=b)(x), y)
    numerical = central_diff(loss_plus, loss_minus)
    assert abs(numerical - layer.grad_weights()[0]) < TOL_LINEAR