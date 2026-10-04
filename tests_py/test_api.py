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
