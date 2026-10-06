"""Dropout: train/eval behaviour, inverted scaling, and the backward mask.

The gradient assertions read the stored mask rather than re-deriving it: the
point of ``test_backward_reuses_the_forward_mask`` is that ``backward`` reuses
the *same* random draw as ``forward``. A freshly drawn mask would still pass a
mean-based check while silently breaking every trained network.
"""

import numpy as np

from tests_py._helpers import model_path


def test_eval_is_identity_and_deterministic():
    from core import nn

    layer = nn.Dropout(p=0.5)
    layer.eval()
    x = np.array([1.0, -2.0, 3.0, 0.5, -0.25])

    out = layer(x)
    assert np.allclose(out, x)
    # The point of eval: the same input must give the same answer every time.
    assert np.allclose(layer(x), out)


def test_training_mode_round_trips():
    from core import nn

    layer = nn.Dropout(p=0.5)
    assert layer.training()
    layer.eval()
    assert not layer.training()
    layer.train()
    assert layer.training()


def test_training_drops_roughly_the_requested_fraction():
    from core import nn

    layer = nn.Dropout(p=0.25)
    layer.train()
    out = layer(np.ones(20000))

    dropped = float(np.count_nonzero(out == 0.0)) / out.size
    # Binomial(p=0.25, n=20000) has sd ~0.003, so +-0.02 is many sigma wide.
    assert abs(dropped - 0.25) < 0.02, dropped


def test_inverted_scaling_preserves_the_mean():
    """Survivors are scaled by 1/(1-p) so the expected output equals the input."""
    from core import nn

    layer = nn.Dropout(p=0.5)
    layer.train()
    out = layer(np.full(40000, 3.0))

    assert abs(out.mean() - 3.0) < 0.05, out.mean()


def test_survivors_are_scaled_not_merely_kept():
    """A kept unit must read x/(1-p), so the scale is visible, not implied."""
    from core import nn

    layer = nn.Dropout(p=0.5)
    layer.train()
    out = layer(np.ones(500))

    survivors = out[out != 0.0]
    assert survivors.size > 0
    assert np.allclose(survivors, 2.0), np.unique(survivors)


def test_p_zero_is_the_identity_even_while_training():
    from core import nn

    layer = nn.Dropout(p=0.0)
    layer.train()
    x = np.array([1.0, -2.0, 3.0])
    assert np.allclose(layer(x), x)
    assert np.allclose(layer(x), x)


def test_p_one_drops_everything_while_training_and_nothing_in_eval():
    from core import nn

    layer = nn.Dropout(p=1.0)
    layer.train()
    assert np.allclose(layer(np.ones(10)), np.zeros(10))

    layer.eval()
    assert np.allclose(layer(np.ones(10)), np.ones(10))


def test_p_one_blocks_the_gradient_too():
    """p=1 must scale the gradient by zero, not pass it through untouched."""
    from core import nn

    layer = nn.Dropout(p=1.0)
    layer.train()
    layer(np.ones(4))
    assert np.allclose(layer.backward(np.ones(4)), np.zeros(4))


def test_backward_reuses_the_forward_mask():
    from core import nn

    layer = nn.Dropout(p=0.5)
    layer.train()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = layer(x)

    grad_out = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    grad_in = layer.backward(grad_out)

    # out == x * mask, so the surviving mask values are recoverable from the
    # forward result alone.
    mask = np.where(out != 0.0, out / x, 0.0)
    assert np.allclose(grad_in, grad_out * mask)
    # Dropped units carry no gradient: the observable consequence of the mask.
    assert np.all(grad_in[out == 0.0] == 0.0)


def test_backward_preserves_shape():
    from core import nn

    layer = nn.Dropout(p=0.5)
    layer.train()
    out = layer(np.ones((2, 3, 4)))
    assert out.shape == (2, 3, 4)
    assert layer.backward(np.ones((2, 3, 4))).shape == (2, 3, 4)


def test_backward_in_eval_passes_the_gradient_through():
    from core import nn

    layer = nn.Dropout(p=0.5)
    layer.eval()
    layer(np.ones(4))
    grad_out = np.array([1.0, -2.0, 3.0, -4.0])
    assert np.allclose(layer.backward(grad_out), grad_out)


def test_backward_before_forward_in_eval_is_the_identity():
    """With no mask stored there is nothing to scale by, and that is correct."""
    from core import nn

    layer = nn.Dropout(p=0.5)
    layer.eval()
    grad_out = np.array([1.0, 2.0])
    assert np.allclose(layer.backward(grad_out), grad_out)


def test_rejects_p_outside_the_unit_interval():
    from core import nn

    for bad in (-0.1, -1.0, 1.5, 2.0):
        try:
            nn.Dropout(p=bad)
            raise AssertionError(f"expected ValueError for p={bad}")
        except ValueError as exc:
            assert "[0, 1]" in str(exc), exc


def test_dropout_has_no_parameters():
    from core import nn

    assert list(nn.Dropout(p=0.5).parameters()) == []


def test_functional_dropout_is_the_identity_in_eval():
    from core import nn

    x = np.array([1.0, -2.0, 3.0, 4.0])
    assert np.allclose(nn.functional.dropout(x, p=0.5, training=False), x)


def test_functional_dropout_drops_in_training():
    from core import nn

    out = nn.functional.dropout(np.ones(20000), p=0.25, training=True)
    dropped = float(np.count_nonzero(out == 0.0)) / out.size
    assert abs(dropped - 0.25) < 0.02, dropped


def test_functional_dropout_rejects_bad_p():
    from core import nn

    try:
        nn.functional.dropout(np.ones(4), p=1.5)
        raise AssertionError("expected ValueError for p=1.5")
    except ValueError as exc:
        assert "[0, 1]" in str(exc), exc


def test_dropout_is_saved_with_its_p():
    """p is a hyperparameter, so it has to survive save/load like LeakyReLU's."""
    from core import io, nn

    net = nn.Sequential(
        nn.Linear(2, 2, weights=np.ones(4), bias=np.zeros(2)),
        nn.Dropout(p=0.25),
    )
    x = np.array([0.5, -0.5])

    with model_path() as path:
        io.save_model(net, path)
        reloaded = io.load_model(path)

    assert reloaded.layers[1].p == 0.25

    # Train/eval is runtime state, not architecture: a reloaded model comes back
    # in training mode, so both sides must be put in eval before comparing.
    net.eval()
    reloaded.eval()
    assert np.allclose(reloaded(x), net(x))


def test_dropout_runs_inside_a_sequential_training_loop():
    """The end-to-end shape: a dropout between two layers, trained by hand."""
    from core import losses, nn, optim

    net = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4, 1),
    )
    x = np.array([0.3, -0.2])
    y = np.array([1.0])
    loss_fn = losses.MSELoss()
    opt = optim.SGD(list(net.parameters()), lr=0.05)

    net.train()
    for _ in range(5):
        loss, grad_out = loss_fn(net(x), y)
        grad = grad_out
        for layer in reversed(net.layers):
            if hasattr(layer, "backward"):
                grad = layer.backward(grad)
        opt.step()
        opt.zero_grad()

    assert np.asarray(net(x)).shape == (1,)