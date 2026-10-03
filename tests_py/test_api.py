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
