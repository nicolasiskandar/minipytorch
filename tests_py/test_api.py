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


def test_activation_modules_match_functional():
    from core import nn, activations
    import numpy as np

    x = np.array([1.5, -0.5, 3.0, -7.0])
    assert np.allclose(nn.ReLU()(x), activations.relu(x))
    assert np.allclose(nn.Sigmoid()(x), activations.sigmoid(x))
    assert np.allclose(nn.Tanh()(x), activations.tanh(x))
    assert np.allclose(nn.ReLU6()(x), activations.relu6(x))
    assert np.allclose(nn.LeakyReLU()(x), activations.leaky_relu(x))


def test_leaky_relu_module_stores_alpha():
    from core import nn
    import numpy as np

    layer = nn.LeakyReLU(negative_slope=0.2)
    assert layer.negative_slope == 0.2
    out = layer(np.array([-4.0, 2.0]))
    assert np.allclose(out, np.array([-0.8, 2.0]))


def test_activation_modules_in_sequential_have_no_parameters():
    from core import nn
    import numpy as np

    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1), nn.Sigmoid())
    assert len(list(net.parameters())) == 4
    assert np.asarray(net(np.array([0.5, -0.5]))).shape == (1,)


def test_flatten_flattens_to_one_dimension():
    from core import nn
    import numpy as np

    out = nn.Flatten()(np.arange(24, dtype=float).reshape(2, 3, 4))
    assert out.shape == (24,)
    assert np.allclose(out, np.arange(24, dtype=float))


def test_conv2d_forward_shape_and_hand_computed_value():
    from core import nn
    import numpy as np

    conv = nn.Conv2d(1, 1, 2, weights=[1.0, 1.0, 1.0, 1.0])
    conv._bias = np.zeros(1)
    out = conv(np.arange(9, dtype=float).reshape(1, 3, 3))
    assert out.shape == (1, 2, 2)
    assert np.allclose(out, np.array([[[8.0, 12.0], [20.0, 24.0]]]))


def test_conv2d_stride_and_padding_shapes():
    from core import nn
    import numpy as np

    x = np.zeros((1, 5, 5))
    assert nn.Conv2d(1, 1, 3)(x).shape == (1, 3, 3)
    assert nn.Conv2d(1, 1, 3, stride=2)(x).shape == (1, 2, 2)
    assert nn.Conv2d(1, 1, 3, padding=1)(x).shape == (1, 5, 5)


def test_conv2d_gradient_numerical():
    from core import nn, losses
    import numpy as np

    np.random.seed(3)
    conv = nn.Conv2d(2, 2, 3, padding=1)
    x = np.random.randn(2, 4, 4)
    y = np.random.randn(2, 4, 4)
    loss_fn = losses.MSELoss()

    _loss, grad_out = loss_fn(conv(x), y)
    grad_in = conv.backward(grad_out)

    w = conv._weights.copy()
    eps = 1e-6
    for idx in [(0, 0, 0, 0), (1, 1, 2, 2)]:
        w_plus = w.copy()
        w_plus[idx] += eps
        w_minus = w.copy()
        w_minus[idx] -= eps
        plus = nn.Conv2d(2, 2, 3, padding=1, weights=w_plus.reshape(-1))
        plus._bias = conv._bias.copy()
        minus = nn.Conv2d(2, 2, 3, padding=1, weights=w_minus.reshape(-1))
        minus._bias = conv._bias.copy()
        lp, _ = loss_fn(plus(x), y)
        lm, _ = loss_fn(minus(x), y)
        assert abs((lp - lm) / (2 * eps) - conv._grad_weights[idx]) < 1e-5, idx

    x_plus = x.copy()
    x_plus[0, 1, 1] += eps
    x_minus = x.copy()
    x_minus[0, 1, 1] -= eps
    lp, _ = loss_fn(conv(x_plus), y)
    lm, _ = loss_fn(conv(x_minus), y)
    assert abs((lp - lm) / (2 * eps) - grad_in[0, 1, 1]) < 1e-5


def test_conv2d_parameters_expose_weight_and_bias():
    from core import nn, optim
    import numpy as np

    conv = nn.Conv2d(1, 2, 3)
    named = dict((n, p) for n, p in conv.named_parameters())
    assert set(named) == {"weight", "bias"}
    assert named["weight"].value.shape == (2, 1, 3, 3)
    assert named["bias"].value.shape == (2,)

    out = conv(np.ones((1, 5, 5)))
    _loss, g = __import__("core.losses", fromlist=["MSELoss"]).MSELoss()(
        out, np.zeros_like(out)
    )
    conv.backward(g)
    before = conv._weights.copy()
    optim.SGD(list(conv.parameters()), lr=0.1).step()
    assert not np.allclose(conv._weights, before)


def test_conv2d_rejects_wrong_rank_input():
    from core import nn
    import numpy as np

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


def test_conv2d_then_flatten_then_linear_end_to_end():
    from core import nn, losses, optim
    import numpy as np

    net = nn.Sequential(
        nn.Conv2d(1, 2, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2 * 2 * 2, 1),
        nn.Sigmoid(),
    )
    x = np.random.randn(1, 4, 4)
    y = np.array([1.0])
    loss_fn = losses.MSELoss()
    opt = optim.SGD(list(net.parameters()), lr=0.05)

    first_loss = None
    for _ in range(5):
        pred = net(x)
        loss, grad_out = loss_fn(pred, y)
        grad = grad_out
        for layer in reversed(net.layers):
            if hasattr(layer, "backward"):
                grad = layer.backward(grad)
        opt.step()
        opt.zero_grad()
        if first_loss is None:
            first_loss = loss
    assert net(x).shape == (1,)


def test_maxpool2d_picks_window_maximum():
    from core import nn
    import numpy as np

    x = np.arange(16, dtype=float).reshape(1, 4, 4)
    out = nn.MaxPool2d(2)(x)
    assert out.shape == (1, 2, 2)
    assert np.allclose(out, np.array([[[5.0, 7.0], [13.0, 15.0]]]))


def test_avgpool2d_averages_window():
    from core import nn
    import numpy as np

    x = np.arange(16, dtype=float).reshape(1, 4, 4)
    out = nn.AvgPool2d(2)(x)
    assert np.allclose(out, np.array([[[2.5, 4.5], [10.5, 12.5]]]))


def test_pooling_strides():
    from core import nn
    import numpy as np

    x = np.zeros((1, 5, 5))
    assert nn.MaxPool2d(2)(x).shape == (1, 2, 2)
    assert nn.MaxPool2d(2, stride=1)(x).shape == (1, 4, 4)
    assert nn.AvgPool2d(3, stride=1)(x).shape == (1, 3, 3)


def test_pooling_gradients_numerical():
    from core import nn, losses
    import numpy as np

    np.random.seed(5)
    x = np.random.randn(2, 4, 4)
    y = np.random.randn(2, 2, 2)
    loss_fn = losses.MSELoss()

    for pool_cls in (nn.MaxPool2d, nn.AvgPool2d):
        pool = pool_cls(2)
        _loss, grad_out = loss_fn(pool(x), y)
        grad_in = pool.backward(grad_out)
        eps = 1e-6
        for pos in [(0, 0, 0), (1, 2, 3)]:
            x_plus = x.copy()
            x_plus[pos] += eps
            x_minus = x.copy()
            x_minus[pos] -= eps
            lp, _ = loss_fn(pool_cls(2)(x_plus), y)
            lm, _ = loss_fn(pool_cls(2)(x_minus), y)
            assert abs((lp - lm) / (2 * eps) - grad_in[pos]) < 1e-5, (pool_cls, pos)


def test_pooling_rejects_input_smaller_than_kernel():
    from core import nn
    import numpy as np

    try:
        nn.MaxPool2d(5)(np.zeros((1, 3, 3)))
        raise AssertionError("expected ValueError for oversized kernel")
    except ValueError:
        pass


def test_adam_matches_reference_formula():
    from core import nn, optim
    import numpy as np

    class FakeParam:
        kind = "weight"
        layer = None

        def __init__(self, value, grad):
            self._value = np.array([value], dtype=float)
            self._grad = np.array([grad], dtype=float)

        @property
        def value(self):
            return self._value

        def grad(self):
            return self._grad

        def set_value(self, values):
            self._value = np.asarray(values, dtype=float)
            return self

        def zero_grad(self):
            self._grad = np.zeros_like(self._grad)

        def apply_gradients(self, lr, momentum):
            return self

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
    from core import nn, optim
    import numpy as np

    class FakeParam:
        kind = "weight"
        layer = None

        def __init__(self, value, grad):
            self._value = np.array([value], dtype=float)
            self._grad = np.array([grad], dtype=float)

        @property
        def value(self):
            return self._value

        def grad(self):
            return self._grad

        def set_value(self, values):
            self._value = np.asarray(values, dtype=float)
            return self

        def zero_grad(self):
            self._grad = np.zeros_like(self._grad)

        def apply_gradients(self, lr, momentum):
            return self

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
    import numpy as np

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


def test_save_load_roundtrip_conv_pool_network():
    from core import nn, io
    import numpy as np
    import tempfile
    import os

    net = nn.Sequential(
        nn.Conv2d(1, 2, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8, 3),
        nn.Tanh(),
        nn.Linear(3, 1),
        nn.Sigmoid(),
    )
    x = np.random.RandomState(0).randn(1, 4, 4)
    expected = net(x)

    path = tempfile.mktemp()
    try:
        io.save_model(net, path)
        reloaded = io.load_model(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    assert len(reloaded.layers) == len(net.layers)
    assert np.allclose(reloaded(x), expected)


def test_save_load_preserves_conv_hyperparameters():
    from core import nn, io
    import numpy as np
    import tempfile
    import os

    net = nn.Sequential(
        nn.Conv2d(2, 3, 3, stride=2, padding=1), nn.AvgPool2d(2, stride=1)
    )
    x = np.random.RandomState(1).randn(2, 6, 6)
    expected = net(x)

    path = tempfile.mktemp()
    try:
        io.save_model(net, path)
        reloaded = io.load_model(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    conv = reloaded.layers[0]
    assert conv.in_channels == 2
    assert conv.out_channels == 3
    assert conv.kernel_size == (3, 3)
    assert conv.stride == (2, 2)
    assert conv.padding == (1, 1)
    assert reloaded.layers[1].kernel_size == (2, 2)
    assert reloaded.layers[1].stride == (1, 1)
    assert np.allclose(reloaded(x), expected)


def test_save_load_preserves_leaky_relu_slope():
    from core import nn, io
    import numpy as np
    import tempfile
    import os

    net = nn.Sequential(nn.Linear(2, 2), nn.LeakyReLU(negative_slope=0.25))
    x = np.array([1.0, -2.0])
    expected = net(x)

    path = tempfile.mktemp()
    try:
        io.save_model(net, path)
        reloaded = io.load_model(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    assert reloaded.layers[1].negative_slope == 0.25
    assert np.allclose(reloaded(x), expected)


def test_load_model_rejects_unknown_layer_tag():
    from core import io
    import tempfile
    import os

    path = tempfile.mktemp()
    try:
        with open(path, "w", encoding="utf-8") as out:
            out.write("minipytorch-v2\n1\nbanana 0\n")
        try:
            io.load_model(path)
            raise AssertionError("expected ValueError for unknown layer tag")
        except ValueError:
            pass
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_load_model_rejects_inconsistent_conv_weight_count():
    from core import io
    import tempfile
    import os

    path = tempfile.mktemp()
    try:
        with open(path, "w", encoding="utf-8") as out:
            out.write("minipytorch-v2\n1\nconv2d 1 2 3 3 1 0 1\n17 2\n")
            out.write(" ".join(["0.0"] * 17) + "\n")
            out.write("0.0 0.0\n")
        try:
            io.load_model(path)
            raise AssertionError("expected ValueError for wrong weight count")
        except ValueError:
            pass
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_activation_modules_have_backward():
    from core import nn
    import numpy as np

    for cls in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.ReLU6, nn.LeakyReLU):
        layer = cls()
        out = layer(np.array([0.5, -1.5]))
        grad = layer.backward(np.array([1.0, 1.0]))
        assert np.asarray(grad).shape == (2,), cls.__name__


def test_relu_backward_blocks_negative_pre_activation():
    from core import nn, losses
    import numpy as np

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


def test_sigmoid_backward_uses_output_derivative():
    from core import nn
    import numpy as np

    act = nn.Sigmoid()
    x = np.array([0.0, 1.0, -1.0])
    out = act(x)
    grad = act.backward(np.ones(3))
    assert np.allclose(grad, out * (1.0 - out))


def test_tanh_backward_uses_output_derivative():
    from core import nn
    import numpy as np

    act = nn.Tanh()
    x = np.array([0.0, 1.0, -1.0])
    out = act(x)
    grad = act.backward(np.ones(3))
    assert np.allclose(grad, 1.0 - out * out)


def test_leaky_relu_module_backward_uses_slope():
    from core import nn
    import numpy as np

    act = nn.LeakyReLU(negative_slope=0.25)
    act(np.array([-4.0, 2.0]))
    grad = act.backward(np.ones(2))
    assert np.allclose(grad, np.array([0.25, 1.0]))


def test_relu6_backward_is_zero_outside_range():
    from core import nn
    import numpy as np

    act = nn.ReLU6()
    act(np.array([-1.0, 3.0, 9.0]))
    grad = act.backward(np.ones(3))
    assert np.allclose(grad, np.array([0.0, 1.0, 0.0]))


def test_activation_backward_before_forward_raises():
    from core import nn
    import numpy as np

    try:
        nn.ReLU().backward(np.ones(2))
        raise AssertionError("expected RuntimeError before any forward")
    except RuntimeError:
        pass


def test_relu_backward_gate_matches_numeric_gradient():
    from core import nn, losses
    import numpy as np

    # Drive the pre-activation negative by bias alone, then confirm the loss is
    # insensitive to the weights (ReLU output is pinned at zero) while a
    # positive pre-activation passes the gradient through.
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
