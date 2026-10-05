"""Activation functions and activation modules (forward and backward)."""

import numpy as np


def test_leaky_relu_activation():
    from core import activations, nn

    assert abs(activations.leaky_relu(-4.0) - (-0.04)) < 1e-12
    assert abs(activations.leaky_relu(4.0) - 4.0) < 1e-12
    assert abs(activations.leaky_relu_deriv(4.0) - 1.0) < 1e-12
    assert abs(activations.leaky_relu_deriv(-4.0) - 0.01) < 1e-12

    layer = nn.Linear(2, 3, activation='leaky_relu', weights=[1, 2, 3, 4, 5, 6], bias=[0, 0, 0])
    out = layer(np.array([-1.0, 0.0]))
    expected = activations.leaky_relu(np.array([-1.0, -3.0, -5.0]))
    assert np.allclose(out, expected)


def test_leaky_relu_custom_alpha():
    from core import activations

    assert abs(float(activations.leaky_relu(-4.0, alpha=0.2)) - (-0.8)) < 1e-12
    assert abs(float(activations.leaky_relu(4.0, alpha=0.2)) - 4.0) < 1e-12
    assert abs(float(activations.leaky_relu_deriv(4.0, alpha=0.2)) - 1.0) < 1e-12
    assert abs(float(activations.leaky_relu_deriv(-4.0, alpha=0.2)) - 0.2) < 1e-12

    values = activations.leaky_relu(np.array([-4.0, 4.0]), alpha=0.2)
    assert np.allclose(values, np.array([-0.8, 4.0]))
    derivs = activations.leaky_relu_deriv(np.array([-4.0, 4.0]), alpha=0.2)
    assert np.allclose(derivs, np.array([0.2, 1.0]))


def test_leaky_relu_default_alpha_uses_asm_path():
    from core import activations

    assert isinstance(activations.leaky_relu(-4.0), float)
    assert isinstance(activations.leaky_relu(4.0), float)
    assert isinstance(activations.leaky_relu_deriv(-4.0), float)
    assert isinstance(activations.leaky_relu_deriv(4.0), float)


def test_relu6_layer_and_activation_kinds():
    from core import activations, nn

    x = np.array([-1.0, 0.0, 3.0, 10.0])
    assert np.allclose(activations.relu6(x), [0.0, 0.0, 3.0, 6.0])
    assert np.allclose(activations.relu6_deriv(x), [0.0, 0.0, 1.0, 0.0])

    # activation strings must map to the distinct kinds, not fall back to ReLU
    assert nn.Linear(2, 2, activation='relu6').activation_kind == nn.ActivationKind.RELU6
    assert nn.Linear(2, 2, activation='leaky_relu').activation_kind == nn.ActivationKind.LEAKYRELU
    assert nn.Linear(2, 2, activation='relu').activation_kind == nn.ActivationKind.RELU

    layer = nn.Linear(2, 2, activation='relu6', weights=[1, 0, 0, 1], bias=[0, 0])
    assert np.allclose(layer(np.array([9.0, 1.0])), [6.0, 1.0])


def test_activation_modules_match_functional():
    from core import nn, activations

    x = np.array([1.5, -0.5, 3.0, -7.0])
    assert np.allclose(nn.ReLU()(x), activations.relu(x))
    assert np.allclose(nn.Sigmoid()(x), activations.sigmoid(x))
    assert np.allclose(nn.Tanh()(x), activations.tanh(x))
    assert np.allclose(nn.ReLU6()(x), activations.relu6(x))
    assert np.allclose(nn.LeakyReLU()(x), activations.leaky_relu(x))


def test_leaky_relu_module_stores_alpha():
    from core import nn

    layer = nn.LeakyReLU(negative_slope=0.2)
    assert layer.negative_slope == 0.2
    out = layer(np.array([-4.0, 2.0]))
    assert np.allclose(out, np.array([-0.8, 2.0]))


def test_activation_modules_in_sequential_have_no_parameters():
    from core import nn

    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1), nn.Sigmoid())
    assert len(list(net.parameters())) == 4
    assert np.asarray(net(np.array([0.5, -0.5]))).shape == (1,)


def test_activation_modules_have_backward():
    from core import nn

    for cls in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.ReLU6, nn.LeakyReLU):
        layer = cls()
        out = layer(np.array([0.5, -1.5]))
        grad = layer.backward(np.array([1.0, 1.0]))
        assert np.asarray(grad).shape == (2,), cls.__name__


def test_activation_backward_before_forward_raises():
    from core import nn

    try:
        nn.ReLU().backward(np.ones(2))
        raise AssertionError("expected RuntimeError before any forward")
    except RuntimeError:
        pass


def test_sigmoid_backward_uses_output_derivative():
    from core import nn

    act = nn.Sigmoid()
    x = np.array([0.0, 1.0, -1.0])
    out = act(x)
    grad = act.backward(np.ones(3))
    assert np.allclose(grad, out * (1.0 - out))


def test_tanh_backward_uses_output_derivative():
    from core import nn

    act = nn.Tanh()
    x = np.array([0.0, 1.0, -1.0])
    out = act(x)
    grad = act.backward(np.ones(3))
    assert np.allclose(grad, 1.0 - out * out)


def test_leaky_relu_module_backward_uses_slope():
    from core import nn

    act = nn.LeakyReLU(negative_slope=0.25)
    act(np.array([-4.0, 2.0]))
    grad = act.backward(np.ones(2))
    assert np.allclose(grad, np.array([0.25, 1.0]))


def test_relu6_backward_is_zero_outside_range():
    from core import nn

    act = nn.ReLU6()
    act(np.array([-1.0, 3.0, 9.0]))
    grad = act.backward(np.ones(3))
    assert np.allclose(grad, np.array([0.0, 1.0, 0.0]))


def test_relu_backward_blocks_negative_pre_activation():
    """A dead unit must not pass the loss gradient through to the weights.

    This is the regression guard for the class of bug where a module in a
    Sequential is skipped by the manual backward loop because it lacks
    ``backward``: training looks fine and the weights never move.
    """
    from core import nn, losses

    conv = nn.Conv2d(1, 1, 1, weights=[1.0])
    conv._bias = np.zeros(1)
    relu = nn.ReLU()
    loss_fn = losses.MSELoss()

    out = relu(conv(np.array([[[-1.0]]])))
    assert np.allclose(out, 0.0)
    _loss, grad_out = loss_fn(out, np.ones((1, 1, 1)))
    grad_relu = relu.backward(grad_out)
    assert np.allclose(grad_relu, 0.0), "ReLU derivative must gate the gradient"
    conv.backward(grad_relu)
    assert np.allclose(conv._grad_weights, 0.0)


def test_relu_backward_gate_matches_numeric_gradient():
    """Drive the pre-activation sign by bias alone, then confirm the loss is
    insensitive to the weights when ReLU is dead and that the gradient flows
    when it is live.
    """
    from core import nn, losses

    loss_fn = losses.MSELoss()

    killed = nn.Conv2d(1, 1, 3, padding=0, weights=[0.5] * 9)
    killed._bias = np.array([-10.0])
    x = np.ones((1, 3, 3))
    y = np.array([[[0.25]]])
    act = nn.ReLU()
    _loss, grad_out = loss_fn(act(killed(x)), y)
    gate = act.backward(grad_out)
    assert np.allclose(gate, 0.0)
    killed.backward(gate)
    assert np.allclose(killed._grad_weights, 0.0)

    live = nn.Conv2d(1, 1, 3, padding=0, weights=[0.5] * 9)
    live._bias = np.array([10.0])
    _loss, grad_out = loss_fn(act(live(x)), y)
    gate = act.backward(grad_out)
    assert np.allclose(gate, grad_out)
    live.backward(gate)
    assert not np.allclose(live._grad_weights, 0.0)