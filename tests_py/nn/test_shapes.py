"""Conv -> ReLU -> MaxPool -> Flatten -> Linear, exercising every handoff.

Kept separate from test_end_to_end.py because the point here is shape
agreement between adjacent layers, not the training loop.
"""

import numpy as np


def _feature_width():
    """The flattened width this network actually produces."""
    from core import nn

    x = np.zeros((1, 4, 4))
    h = nn.Conv2d(1, 2, 3)(x)
    h = nn.ReLU()(h)
    h = nn.MaxPool2d(2)(h)
    return int(np.prod(nn.Flatten()(h).shape))


def test_flattened_width_is_two_not_eight():
    """Regression guard for the width bug.

    Conv 3x3 (no padding) on 4x4 leaves a 2x2 map per channel. MaxPool2d(2)
    reduces 2x2 to 1x1, so the flattened width is the channel count, 2.

    Reading this as ``2 * 2 * 2`` is the natural mistake, and it used to be the
    wrong answer: the declared in_features was 8 while the actual activation was
    2 elements wide.
    """
    assert _feature_width() == 2


def test_linear_declared_width_must_match_the_activations():
    from core import nn

    width = _feature_width()
    net = nn.Sequential(
        nn.Conv2d(1, 2, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(width, 1),
    )
    # Runs clean, so the declared width agrees with the real one.
    assert net(np.random.randn(1, 4, 4)).shape == (1,)


def test_mismatched_linear_width_raises_instead_of_reading_memory():
    """The bug this pins: a short input was an out-of-bounds read.

    The native layer takes num_inputs doubles from whatever buffer it is given,
    so declaring 8 inputs against a 2-element activation read past the end and
    produced ~1e251 gradients and NaN losses instead of an error.
    """
    from core import nn

    layer = nn.Linear(8, 1, weights=np.zeros(8), bias=np.zeros(1))
    try:
        layer(np.zeros(2))
        raise AssertionError("expected ValueError on a short input")
    except ValueError as exc:
        assert "8" in str(exc) and "2" in str(exc), exc


def test_gradients_stay_finite_when_widths_agree():
    """After the fix, gradients are bounded by the inputs, not memory junk."""
    from core import losses, nn

    net = nn.Sequential(
        nn.Conv2d(1, 2, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(_feature_width(), 1, activation='relu',
                  weights=np.ones(2), bias=np.array([0.1])),
    )
    x = np.random.randn(1, 4, 4)
    loss, grad_out = losses.MSELoss()(net(x), np.array([1.0]))

    assert np.isfinite(loss)
    grad = grad_out
    for layer in reversed(net.layers):
        if hasattr(layer, "backward"):
            grad = layer.backward(grad)
    assert np.all(np.isfinite(grad))
    # A sanity bound: with unit weights and unit inputs, no gradient entry
    # should exceed a small multiple of the number of terms contributing.
    assert np.max(np.abs(net.layers[4].grad_weights())) < 1e3


def test_linear_backward_before_forward_raises():
    """Without a forward there is no cached input, so the read would be wild."""
    from core import nn

    try:
        nn.Linear(2, 1).backward(np.array([1.0]))
        raise AssertionError("expected RuntimeError before any forward")
    except RuntimeError as exc:
        assert "before any forward" in str(exc), exc


def test_linear_backward_rejects_a_wrong_sized_gradient():
    from core import nn

    layer = nn.Linear(2, 1)
    layer(np.array([1.0, 2.0]))
    try:
        layer.backward(np.ones(3))
        raise AssertionError("expected ValueError on a wrong-sized gradient")
    except ValueError as exc:
        assert "gradient" in str(exc), exc