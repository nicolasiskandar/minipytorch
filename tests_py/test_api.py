import numpy as np
import sys
sys.path.insert(0, '/home/nicolas/workspace/minipytorch')

from core import nn, losses, activations


def test_linear_forward():
    l = nn.Linear(2, 3)
    out = l.forward(np.array([1.0, 2.0]))
    assert out.shape == (3,)


def test_mse_loss():
    loss_fn = losses.MSELoss()
    loss, grad = loss_fn(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert abs(loss - 1.0) < 1e-10
    assert grad.shape == (2,)


def test_backward_shapes():
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
    import numpy as np
    l = nn.Linear(2, 3, weights=[1,2,3,4,5,6], bias=[0.1,0.2,0.3])
    w = np.array(l.weights())
    assert w.shape == (6,)
    assert abs(w[0] - 1.0) < 1e-12
    l.set_weights([0,1,2,3,4,5])
    w2 = np.array(l.weights())
    assert abs(w2[0] - 0.0) < 1e-12


def test_sequential():
    from core import nn
    import numpy as np
    l1 = nn.Linear(2, 2, activation='tanh')
    l2 = nn.Linear(2, 1, activation='sigmoid')
    seq = nn.Sequential(l1, l2)
    out = seq(np.array([0.5, -0.5]))
    assert out.shape == (1,)


def test_bce_loss():
    from core import losses
    import numpy as np
    loss_fn = losses.BCELoss()
    # simple case: pred 0.5, target 1.0 -> mean BCE is -log(0.5)
    loss, grad = loss_fn(np.array([0.5]), np.array([1.0]))
    assert abs(loss - (-np.log(0.5))) < 1e-10
    assert grad.shape == (1,)


def test_module_parameters():
    from core import nn
    import numpy as np
    l1 = nn.Linear(2, 3)
    l2 = nn.Linear(3, 1)
    seq = nn.Sequential(l1, l2)
    params = list(seq.parameters())
    assert len(params) == 4  # 2 weight+bias pairs
    named = list(seq.named_parameters())
    assert len(named) == 4
    assert named[0][0].startswith('0.weight') or named[0][0] == '0.weight'


def test_numerical_gradients_mse():
    from core import nn, losses
    import numpy as np
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
    eps = 1e-6
    w = np.array(l.weights(), dtype=float)
    # perturb w[0]
    w1 = w.copy(); w1[0] += eps
    l2 = nn.Linear(2, 2, activation='tanh', weights=w1, bias=np.array([0.0,0.0]))
    loss1, _ = loss_fn(l2(x), y)
    w2 = w.copy(); w2[0] -= eps
    l3 = nn.Linear(2, 2, activation='tanh', weights=w2, bias=np.array([0.0,0.0]))
    loss2, _ = loss_fn(l3(x), y)
    num_grad = (loss1 - loss2) / (2*eps)
    anal = np.array(l.grad_weights())[0]
    assert abs(num_grad - anal) < 1e-4


def test_numerical_gradients_bce():
    from core import nn, losses
    import numpy as np
    np.random.seed(42)
    l = nn.Linear(2, 1, activation='sigmoid', weights=np.array([0.2, -0.3]), bias=np.array([0.1]))
    x = np.array([0.7, 0.4])
    y = np.array([1.0])
    loss_fn = losses.BCELoss()
    h = l(x)
    loss, g = loss_fn(h, y)
    _ = l.backward(g)
    eps = 1e-6
    w = np.array(l.weights(), dtype=float)
    # w[0]
    w1 = w.copy(); w1[0] += eps
    l2 = nn.Linear(2, 1, activation='sigmoid', weights=w1, bias=np.array([0.1]))
    loss1, _ = loss_fn(l2(x), y)
    w2 = w.copy(); w2[0] -= eps
    l3 = nn.Linear(2, 1, activation='sigmoid', weights=w2, bias=np.array([0.1]))
    loss2, _ = loss_fn(l3(x), y)
    num_grad = (loss1 - loss2) / (2*eps)
    anal = np.array(l.grad_weights())[0]
    assert abs(num_grad - anal) < 1e-4


def test_linear_grad_biases_numerical():
    from core import nn, losses
    import numpy as np
    np.random.seed(7)
    l = nn.Linear(2, 2, activation='tanh', weights=np.array([0.1, -0.2, 0.3, 0.4]), bias=np.array([0.05, -0.1]))
    x = np.array([0.8, -0.3])
    y = np.array([0.4, 0.6])
    loss_fn = losses.MSELoss()
    h = l(x)
    loss, g = loss_fn(h, y)
    _ = l.backward(g)
    eps = 1e-6
    b = np.array(l.biases(), dtype=float)
    # bias[0]
    b1 = b.copy(); b1[0] += eps
    l2 = nn.Linear(2, 2, activation='tanh', weights=l.weights(), bias=b1)
    loss1, _ = loss_fn(l2(x), y)
    b2 = b.copy(); b2[0] -= eps
    l3 = nn.Linear(2, 2, activation='tanh', weights=l.weights(), bias=b2)
    loss2, _ = loss_fn(l3(x), y)
    num_grad = (loss1 - loss2) / (2*eps)
    anal = np.array(l.grad_biases())[0]
    assert abs(num_grad - anal) < 1e-4


def test_sequential_named_params():
    from core import nn
    l1 = nn.Linear(2, 2, activation='tanh')
    l2 = nn.Linear(2, 1, activation='sigmoid')
    seq = nn.Sequential(l1, l2)
    named = list(seq.named_parameters())
    assert len(named) == 4
    names = [n for n, _ in named]
    assert all('.' in n for n in names)


def test_linear_activation_parity_shapes():
    from core import nn
    import numpy as np
    x = np.array([1.0, -0.5])
    for act in ['sigmoid', 'tanh', 'relu']:
        l = nn.Linear(2, 3, activation=act)
        out = l(x)
        assert out.shape == (3,)


def test_save_load_roundtrip():
    from core import nn, io
    import numpy as np
    import tempfile
    import os

    l1 = nn.Linear(2, 3, activation='tanh')
    l2 = nn.Linear(3, 1, activation='sigmoid')
    seq = nn.Sequential(l1, l2)
    x = np.array([0.5, -0.3])
    out1 = seq(x)
    f = tempfile.mktemp()
    try:
        io.save_model(seq, f)
        seq2 = io.load_model(f)
        out2 = seq2(x)
        assert np.allclose(out1, out2)
    finally:
        if os.path.exists(f):
            os.remove(f)


def test_leaky_relu_activation():
    from core import activations, nn
    import numpy as np

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
    import numpy as np

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


def test_leaky_relu_gradient_numerical():
    from core import nn, losses
    import numpy as np

    w = np.array([0.2, -0.3, 0.1, 0.4])
    b = np.array([0.05, -0.1])
    x = np.array([0.8, -0.3])
    y = np.array([0.4, 0.6])
    loss_fn = losses.MSELoss()

    layer = nn.Linear(2, 2, activation='leaky_relu', weights=w, bias=b)
    out = layer(x)
    _loss, grad_out = loss_fn(out, y)
    _ = layer.backward(grad_out)

    eps = 1e-6
    w_plus = w.copy()
    w_plus[0] += eps
    w_minus = w.copy()
    w_minus[0] -= eps
    loss_plus, _ = loss_fn(nn.Linear(2, 2, activation='leaky_relu', weights=w_plus, bias=b)(x), y)
    loss_minus, _ = loss_fn(nn.Linear(2, 2, activation='leaky_relu', weights=w_minus, bias=b)(x), y)
    numerical = (loss_plus - loss_minus) / (2 * eps)
    assert abs(numerical - layer.grad_weights()[0]) < 1e-4


def test_relu6_layer_and_activation_kinds():
    from core import activations, nn
    import numpy as np

    x = np.array([-1.0, 0.0, 3.0, 10.0])
    assert np.allclose(activations.relu6(x), [0.0, 0.0, 3.0, 6.0])
    assert np.allclose(activations.relu6_deriv(x), [0.0, 0.0, 1.0, 0.0])

    # activation strings must map to the distinct kinds, not fall back to ReLU
    assert nn.Linear(2, 2, activation='relu6').activation_kind == nn.ActivationKind.RELU6
    assert nn.Linear(2, 2, activation='leaky_relu').activation_kind == nn.ActivationKind.LEAKYRELU
    assert nn.Linear(2, 2, activation='relu').activation_kind == nn.ActivationKind.RELU

    layer = nn.Linear(2, 2, activation='relu6', weights=[1, 0, 0, 1], bias=[0, 0])
    assert np.allclose(layer(np.array([9.0, 1.0])), [6.0, 1.0])


def test_sgd_step_updates_weights():
    from core import nn, optim, losses
    import numpy as np

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
    import numpy as np

    lin = nn.Linear(2, 2, weights=[1.0, 2.0, 3.0, 4.0], bias=[0.5, -0.5])
    before = lin.weights().copy()
    optim.SGD(lin.parameters(), lr=0.1).step()
    assert np.allclose(lin.weights(), before)


def test_sgd_momentum_matches_formula():
    from core import nn, optim, losses
    import numpy as np

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


def test_zero_grad_clears_gradients():
    from core import nn, losses
    import numpy as np

    lin = nn.Linear(2, 2, weights=[1.0, 2.0, 3.0, 4.0], bias=[0.5, -0.5])
    out = lin.forward(np.array([1.0, 1.0]))
    _loss, grad_out = losses.MSELoss()(out, np.array([0.5, 0.5]))
    lin.backward(grad_out)
    assert any(p.grad().any() for p in lin.parameters())

    lin.zero_grad()
    assert all(not p.grad().any() for p in lin.parameters())


def test_sgd_zero_grad_reaches_params():
    from core import nn, optim, losses
    import numpy as np

    lin = nn.Linear(2, 2, weights=[1.0, 2.0, 3.0, 4.0], bias=[0.5, -0.5])
    out = lin.forward(np.array([1.0, 1.0]))
    _loss, grad_out = losses.MSELoss()(out, np.array([0.5, 0.5]))
    lin.backward(grad_out)

    opt = optim.SGD(lin.parameters(), lr=0.1, momentum=0.9)
    opt.zero_grad()
    assert all(not p.grad().any() for p in lin.parameters())


def test_sgd_rejects_non_parameter_params():
    from core import nn, optim
    import numpy as np

    lin = nn.Linear(2, 2)
    try:
        optim.SGD([np.zeros(4), np.zeros(2)], lr=0.1)
        raise AssertionError("expected TypeError for plain arrays")
    except TypeError:
        pass


def test_parameters_yield_stable_handles():
    from core import nn

    lin = nn.Linear(2, 2)
    first = list(lin.parameters())
    second = list(lin.parameters())
    assert all(a is b for a, b in zip(first, second))
    assert [p.kind for p in first] == ['weight', 'bias']
    assert all(p.layer is lin for p in first)


def test_save_load_roundtrip_all_activations():
    from core import nn, io
    import numpy as np
    import tempfile
    import os

    x = np.array([0.5, -0.3])
    for act in ['sigmoid', 'tanh', 'relu', 'relu6', 'leaky_relu']:
        l1 = nn.Linear(2, 3, activation=act)
        l2 = nn.Linear(3, 1, activation=act)
        seq = nn.Sequential(l1, l2)
        expected = seq(x)

        f = tempfile.mktemp()
        try:
            io.save_model(seq, f)
            reloaded = io.load_model(f)
        finally:
            if os.path.exists(f):
                os.remove(f)

        assert int(reloaded.layers[0].activation_kind) == int(l1.activation_kind), act
        assert int(reloaded.layers[1].activation_kind) == int(l2.activation_kind), act
        assert np.allclose(reloaded(x), expected), act


def test_save_model_writes_activation_name_verbatim():
    from core import nn, io
    import tempfile
    import os

    for act in ['relu6', 'leaky_relu']:
        seq = nn.Sequential(nn.Linear(2, 2, activation=act))
        f = tempfile.mktemp()
        try:
            io.save_model(seq, f)
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()
        finally:
            if os.path.exists(f):
                os.remove(f)
        assert act in text, act
        assert "sigmoid" not in text, act


def test_load_model_rejects_unknown_activation():
    from core import io
    import tempfile
    import os

    f = tempfile.mktemp()
    try:
        with open(f, "w", encoding="utf-8") as out:
            out.write("1\n2\ngelu 0.5 1.0 2.0\ngelu 0.5 3.0 4.0\n")
        try:
            io.load_model(f)
            raise AssertionError("expected ValueError for unknown activation")
        except ValueError:
            pass
    finally:
        if os.path.exists(f):
            os.remove(f)


def test_load_model_rejects_malformed_bias():
    from core import io
    import tempfile
    import os

    f = tempfile.mktemp()
    try:
        with open(f, "w", encoding="utf-8") as out:
            out.write("1\n1\nrelu notanumber 1.0 2.0\n")
        try:
            io.load_model(f)
            raise AssertionError("expected ValueError for malformed bias")
        except ValueError:
            pass
    finally:
        if os.path.exists(f):
            os.remove(f)
