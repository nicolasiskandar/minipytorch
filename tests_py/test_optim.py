"""Optimizers: SGD (plain, momentum) and Adam."""

import numpy as np

from tests_py._helpers import FakeParam


def test_sgd_step_updates_weights():
    from core import nn, optim, losses

    lin = nn.Linear(2, 2, weights=[1.0, 2.0, 3.0, 4.0], bias=[0.5, -0.5])
    out = lin.forward(np.array([1.0, 1.0]))
    _loss, grad_out = losses.MSELoss()(out, np.array([0.5, 0.5]))
    lin.backward(grad_out)

    before = lin.weights().copy()
    before_bias = lin.biases().copy()
    optim.SGD(lin.parameters(), lr=0.1).step()
    assert not np.allclose(lin.weights(), before)
    assert not np.allclose(lin.biases(), before_bias)


def test_sgd_step_without_backward_is_a_no_op():
    from core import nn, optim

    lin = nn.Linear(2, 2, weights=[1.0, 2.0, 3.0, 4.0], bias=[0.5, -0.5])
    before = lin.weights().copy()
    optim.SGD(lin.parameters(), lr=0.1).step()
    assert np.allclose(lin.weights(), before)


def test_sgd_momentum_matches_formula():
    from core import nn, optim, losses

    lin = nn.Linear(2, 1, weights=[0.5, -0.25], bias=[0.1])
    out = lin.forward(np.array([1.0, -2.0]))
    _loss, grad_out = losses.MSELoss()(out, np.array([0.5]))
    lin.backward(grad_out)

    lr = 0.1
    momentum = 0.9
    params = list(lin.parameters())
    gradients = [np.array(p.grad(), dtype=float) for p in params]
    velocities = [np.zeros_like(g) for g in gradients]
    expected = [np.array(p.value, dtype=float) for p in params]

    for _ in range(3):
        for i in range(len(params)):
            velocities[i] = momentum * velocities[i] + gradients[i]
            expected[i] = expected[i] - lr * velocities[i]

    opt = optim.SGD(params, lr=lr, momentum=momentum)
    for _ in range(3):
        opt.step()

    for i, p in enumerate(params):
        assert np.allclose(p.value, expected[i], atol=1e-12)


def test_sgd_rejects_non_parameter_params():
    from core import optim

    try:
        optim.SGD([np.zeros(4), np.zeros(2)], lr=0.1)
        raise AssertionError("expected TypeError for plain arrays")
    except TypeError:
        pass


def test_zero_grad_clears_gradients():
    from core import nn, losses

    lin = nn.Linear(2, 2, weights=[1.0, 2.0, 3.0, 4.0], bias=[0.5, -0.5])
    out = lin.forward(np.array([1.0, 1.0]))
    _loss, grad_out = losses.MSELoss()(out, np.array([0.5, 0.5]))
    lin.backward(grad_out)
    assert any(p.grad().any() for p in lin.parameters())

    lin.zero_grad()
    assert all(not p.grad().any() for p in lin.parameters())


def test_sgd_zero_grad_reaches_params():
    from core import nn, optim, losses

    lin = nn.Linear(2, 2, weights=[1.0, 2.0, 3.0, 4.0], bias=[0.5, -0.5])
    out = lin.forward(np.array([1.0, 1.0]))
    _loss, grad_out = losses.MSELoss()(out, np.array([0.5, 0.5]))
    lin.backward(grad_out)

    opt = optim.SGD(lin.parameters(), lr=0.1, momentum=0.9)
    opt.zero_grad()
    assert all(not p.grad().any() for p in lin.parameters())


def test_parameters_yield_stable_handles():
    from core import nn

    lin = nn.Linear(2, 2)
    first = list(lin.parameters())
    second = list(lin.parameters())
    assert all(a is b for a, b in zip(first, second))
    assert [p.kind for p in first] == ['weight', 'bias']
    assert all(p.layer is lin for p in first)


def test_adam_matches_reference_formula():
    from core import optim

    p = FakeParam(1.0, 2.0)
    opt = optim.Adam([p], lr=0.1)
    opt.step()
    m = 0.1 * 2.0
    v = 0.001 * 4.0
    m_hat = m / (1 - 0.9)
    v_hat = v / (1 - 0.999)
    expected = 1.0 - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
    assert abs(p.value[0] - expected) < 1e-12


def test_adam_second_step_uses_bias_correction():
    from core import optim

    p = FakeParam(0.0, 1.0)
    opt = optim.Adam([p], lr=0.1, betas=(0.5, 0.9), eps=1e-8)
    opt.step()
    first = p.value[0]
    opt.step()
    second = p.value[0]

    b1, b2 = 0.5, 0.9
    g = 1.0
    m1 = (1 - b1) * g
    v1 = (1 - b2) * g * g
    step1 = 0.1 * (m1 / (1 - b1)) / (np.sqrt(v1 / (1 - b2)) + 1e-8)
    m2 = b1 * m1 + (1 - b1) * g
    v2 = b2 * v1 + (1 - b2) * g * g
    step2 = 0.1 * (m2 / (1 - b1 ** 2)) / (np.sqrt(v2 / (1 - b2 ** 2)) + 1e-8)
    assert abs(first - (0.0 - step1)) < 1e-12
    assert abs(second - (first - step2)) < 1e-12


def test_adam_reduces_loss_on_a_single_layer():
    from core import nn, optim, losses

    lin = nn.Linear(1, 1, weights=[0.0], bias=[0.0])
    x = np.array([1.0])
    y = np.array([1.0])
    loss_fn = losses.MSELoss()
    opt = optim.Adam(list(lin.parameters()), lr=0.05)

    losses_seen = []
    for _ in range(40):
        loss, g = loss_fn(lin(x), y)
        lin.backward(g)
        losses_seen.append(loss)
        opt.step()
        opt.zero_grad()

    assert losses_seen[-1] < losses_seen[0]
    assert losses_seen[-1] < 1e-2