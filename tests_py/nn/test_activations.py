"""Activation functions and activation modules (forward and backward)."""

import numpy as np

from tests_py._helpers import central_diff, EPS


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

    for cls in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.ReLU6, nn.LeakyReLU, nn.Softmax):
        layer = cls()
        out = layer(np.array([0.5, -1.5]))
        grad = layer.backward(np.ones_like(out))
        assert np.asarray(grad).shape == (2,), cls.__name__


def test_softmax_backward_matches_central_differences():
    """Gradient-check the Jacobian itself, against the function it defines.

    Drives ``L = <g, softmax(x)>`` so the finite difference exercises the same
    quantity ``backward`` is supposed to return.
    """
    from core import nn

    x = np.array([0.7, -1.2, 2.3, 0.1])
    g = np.array([1.0, 2.0, -0.5, 3.0])
    layer = nn.Softmax()
    layer(x)
    analytic = layer.backward(g)

    def total(xx):
        return float(np.sum(g * nn.Softmax()(xx)))

    numeric = np.array([
        central_diff(
            total(x + EPS * np.eye(x.size)[j]),
            total(x - EPS * np.eye(x.size)[j]),
        )
        for j in range(x.size)
    ])
    assert np.allclose(numeric, analytic, atol=1e-6), (numeric, analytic)


def test_softmax_backward_is_the_jacobian_transpose():
    """Closed form: grad = y * (g - dot(y, g)).

    Written out so a wrong-but-plausible variant fails. The two easy mistakes
    are dropping the transpose (``g - y*sum(g)``, which is wrong because it
    substitutes ``sum(g)`` for ``dot(y, g)``) and summing over the wrong axis,
    which silently couples rows of a batch together.
    """
    from core import nn

    x = np.array([0.7, -1.2, 2.3, 0.1])
    g = np.array([1.0, 2.0, -0.5, 3.0])
    layer = nn.Softmax()
    y = layer(x)
    assert np.allclose(layer.backward(g), y * (g - float(np.dot(y, g))))

    # Not the transposed-looking variant, which drops the factor of y.
    assert not np.allclose(layer.backward(g), g - y * g.sum())


def test_softmax_backward_on_a_batch_reduces_per_row():
    """A (N, C) input must reduce along C, never across N.

    Summing over the whole batch would couple samples: one row's gradient would
    depend on another row's incoming gradient.
    """
    from core import nn

    x = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]])
    g = np.array([[1.0, -1.0, 2.0], [3.0, 0.5, -2.0]])
    layer = nn.Softmax()
    y = layer(x)
    grad = layer.backward(g)

    expected = y * (g - np.sum(y * g, axis=1, keepdims=True))
    assert np.allclose(grad, expected)
    # Per-row sums vanish, the signature of the softmax Jacobian's null direction.
    assert np.allclose(grad.sum(axis=1), 0.0)


def test_softmax_backward_honours_a_non_default_dim():
    from core import nn

    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    g = np.ones((3, 2))
    layer = nn.Softmax(dim=0)
    y = layer(x)
    grad = layer.backward(g)

    assert np.allclose(grad, y * (g - np.sum(y * g, axis=0, keepdims=True)))
    # Reductions ran down the columns, so each column sums to zero.
    assert np.allclose(grad.sum(axis=0), 0.0)
    assert not np.allclose(y, nn.Softmax()(x))


def test_softmax_backward_before_forward_raises():
    from core import nn

    try:
        nn.Softmax().backward(np.ones(2))
        raise AssertionError("expected RuntimeError before any forward")
    except RuntimeError as exc:
        assert "before any forward" in str(exc), exc


def test_softmax_in_front_of_cross_entropy_changes_the_answer():
    """Documents a trap: ``CrossEntropyLoss`` already applies softmax.

    ``loss_fn(logits)`` and ``loss_fn(nn.Softmax()(logits))`` are different
    functions -- the second softmaxes twice -- so neither the loss nor the
    gradient matches. Pinned as a divergence rather than an equality so that
    someone "fixing" the apparent inconsistency learns that the composition
    itself is the error.
    """
    from core import losses, nn

    logits = np.array([[1.5, -0.5, 0.25], [0.3, 2.1, -1.4]])
    target = np.array([0, 1])

    loss_fn = losses.CrossEntropyLoss()
    fused_loss, fused_grad = loss_fn(logits, target)

    sm = nn.Softmax()
    split_loss, grad_probs = loss_fn(sm(logits), target)
    split_grad = sm.backward(grad_probs)

    assert abs(split_loss - fused_loss) > 1e-3, "double softmax should differ"
    assert not np.allclose(split_grad, fused_grad)
    # The correct pairing is Linear -> CrossEntropyLoss, with no Softmax layer.
    assert np.allclose(fused_grad, _softmax_minus_onehot(logits, target) / 2)


def _softmax_minus_onehot(logits, target):
    """The gradient CrossEntropyLoss folds in, spelled out independently."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    grad = probs.copy()
    for row, cls in enumerate(target):
        grad[row, int(cls)] -= 1.0
    return grad


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