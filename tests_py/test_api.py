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
