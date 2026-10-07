"""Module/Parameter plumbing, the Sequential container, and Flatten."""

import numpy as np


def test_sequential():
    from core import nn

    l1 = nn.Linear(2, 2, activation='tanh')
    l2 = nn.Linear(2, 1, activation='sigmoid')
    seq = nn.Sequential(l1, l2)
    out = seq(np.array([0.5, -0.5]))
    assert out.shape == (1,)


def test_module_parameters():
    from core import nn

    l1 = nn.Linear(2, 3)
    l2 = nn.Linear(3, 1)
    seq = nn.Sequential(l1, l2)
    params = list(seq.parameters())
    assert len(params) == 4  # 2 weight+bias pairs
    named = list(seq.named_parameters())
    assert len(named) == 4
    assert named[0][0].startswith('0.weight') or named[0][0] == '0.weight'


def test_sequential_named_params():
    from core import nn

    l1 = nn.Linear(2, 2, activation='tanh')
    l2 = nn.Linear(2, 1, activation='sigmoid')
    seq = nn.Sequential(l1, l2)
    named = list(seq.named_parameters())
    assert len(named) == 4
    names = [n for n, _ in named]
    assert all('.' in n for n in names)


def test_flatten_flattens_to_one_dimension():
    from core import nn

    # PyTorch semantics: start_dim=1 (default) preserves dim 0 and flattens
    # everything after it. A (2, 3, 4) volume thus becomes (2, 12), and
    # start_dim=0 collapses the whole thing to a single axis.
    out = nn.Flatten()(np.arange(24, dtype=float).reshape(2, 3, 4))
    assert out.shape == (2, 12)
    assert np.allclose(out, np.arange(24, dtype=float).reshape(2, 12))
    out0 = nn.Flatten(start_dim=0)(np.arange(8, dtype=float).reshape(2, 2, 2))
    assert out0.shape == (8,)


def test_train_and_eval_reach_children():
    """Regression: Module.train recursing is load-bearing for nested layers.

    A container's own flag is useless to the layers inside it. Before this
    recursed, ``Sequential.eval()`` left a nested Dropout training, so
    eval-mode inference returned a different answer for the same input.
    """
    from core import nn

    inner = nn.Dropout(p=0.5)
    seq = nn.Sequential(nn.Linear(2, 2), inner)

    seq.eval()
    assert not seq.training()
    assert not inner.training()

    seq.train()
    assert seq.training()
    assert inner.training()


def test_sequential_eval_makes_forward_deterministic():
    from core import nn

    net = nn.Sequential(
        nn.Linear(2, 8, weights=np.arange(16) * 0.1 - 0.5, bias=np.zeros(8)),
        nn.Dropout(p=0.5),
        nn.Linear(8, 1, weights=np.ones(8) * 0.1, bias=np.zeros(1)),
    )
    x = np.array([0.7, -0.4])

    net.train()
    assert not np.allclose(net(x), net(x)), "training mode should be stochastic"

    net.eval()
    first = net(x)
    assert np.allclose(net(x), first)


def test_zero_grad_reaches_children():
    """Regression: a bare ``return self`` cleared the container's empty grads.

    Nothing owned gradients in the container, so the children kept theirs and
    the next step accumulated onto stale values.
    """
    from core import losses, nn

    net = nn.Sequential(nn.Linear(2, 2, weights=[1.0, 2.0, 3.0, 4.0], bias=[0.0, 0.0]))
    out = net(np.array([1.0, 1.0]))
    _loss, grad_out = losses.MSELoss()(out, np.zeros(2))
    net.layers[0].backward(grad_out)
    assert any(p.grad().any() for p in net.parameters())

    net.zero_grad()
    assert all(not p.grad().any() for p in net.parameters())


def test_train_returns_self_for_chaining():
    from core import nn

    seq = nn.Sequential(nn.Dropout(p=0.5))
    assert seq.train() is seq
    assert seq.eval() is seq
    assert seq.zero_grad() is seq

def test_sequential_backward_rejects_missing_backward():
    """A derivative-less child used to be silently skipped: the gradient
    passed straight through dead units and the model trained on nonsense."""
    import numpy as np

    from core import nn

    class NoBackward(nn.Module):
        def forward(self, x):
            return x

    net = nn.Sequential(nn.Linear(2, 1), NoBackward())
    try:
        net.backward(np.array([1.0]))
        raise AssertionError("expected RuntimeError for child without backward")
    except RuntimeError as e:
        assert "layer 1" in str(e), e
        assert "NoBackward" in str(e), e
