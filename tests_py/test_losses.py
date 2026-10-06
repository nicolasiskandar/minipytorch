"""Loss functions: MSELoss, BCELoss, and CrossEntropyLoss.

CrossEntropyLoss folds softmax into the loss and returns a gradient with
respect to the *logits*, not the probabilities. That pairing is what makes
``Linear -> CrossEntropyLoss`` correct: adding an ``nn.Softmax`` layer in
between would apply softmax twice.
"""

import numpy as np

from tests_py._helpers import central_diff


def test_mse_loss():
    from core import losses

    loss_fn = losses.MSELoss()
    loss, grad = loss_fn(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert abs(loss - 1.0) < 1e-10
    assert grad.shape == (2,)


def test_bce_loss():
    from core import losses

    loss_fn = losses.BCELoss()
    # simple case: pred 0.5, target 1.0 -> mean BCE is -log(0.5)
    loss, grad = loss_fn(np.array([0.5]), np.array([1.0]))
    assert abs(loss - (-np.log(0.5))) < 1e-10
    assert grad.shape == (1,)


def _softmax(logits, axis=-1):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


# The loss guards the log with eps=1e-12, so exact -log(p) is off by ~eps/p.
# 1e-9 keeps the assertions tight enough to catch a wrong formula.
LOG_TOL = 1e-9


def test_cross_entropy_batch_loss_matches_negative_log_softmax():
    from core import losses

    logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    target = np.array([0, 1])

    loss, _grad = losses.CrossEntropyLoss()(logits, target)

    p = _softmax(logits)
    expected = -(np.log(p[0, 0]) + np.log(p[1, 1])) / 2
    assert abs(loss - expected) < LOG_TOL, (loss, expected)


def test_cross_entropy_gradient_is_softmax_minus_onehot_over_batch():
    """The returned gradient is w.r.t. logits, so it sums to zero per row."""
    from core import losses

    logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    _loss, grad = losses.CrossEntropyLoss()(logits, np.array([0, 1]))

    p = _softmax(logits)
    expected = np.stack([
        p[0] - np.array([1.0, 0.0, 0.0]),
        p[1] - np.array([0.0, 1.0, 0.0]),
    ]) / 2
    assert np.allclose(grad, expected)
    assert grad.shape == logits.shape
    # A zero row sum is the signature of the softmax Jacobian's null direction.
    assert np.allclose(grad.sum(axis=1), 0.0)


def test_cross_entropy_ignore_index_skips_the_sample_entirely():
    from core import losses

    logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    _loss, grad = losses.CrossEntropyLoss()(logits, np.array([0, -100]))

    p = _softmax(logits)
    # Ignored rows contribute no loss and no gradient at all. The kept row is
    # the only row left in the mean, so its gradient is unscaled.
    assert np.allclose(grad[1], 0.0)
    assert np.allclose(grad[0], (p[0] - np.array([1.0, 0.0, 0.0])))


def test_cross_entropy_ignore_index_divides_by_kept_count():
    """Ignored rows are dropped from the mean-reduction denominator.

    PyTorch's unweighted 'mean' reduction divides by the number of *valid*
    (non-ignored) targets, so a batch with one ignored target has its losses
    and gradients scaled by 1/kept, not 1/N. Pinning this against the tempting
    alternative -- dividing by the full batch size -- which silently shrinks
    the effective learning rate on batches that contain padding.
    """
    from core import losses

    logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    target = np.array([0, -100])

    loss, grad = losses.CrossEntropyLoss()(logits, target)

    p = _softmax(logits)
    assert abs(loss - (-np.log(p[0, 0]))) < LOG_TOL, loss
    assert np.allclose(grad[0], (p[0] - np.array([1.0, 0.0, 0.0])))


def test_cross_entropy_all_targets_ignored_gives_zero_loss_and_gradient():
    from core import losses

    logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    loss, grad = losses.CrossEntropyLoss()(logits, np.array([-100, -100]))

    assert loss == 0.0
    assert np.allclose(grad, np.zeros_like(logits))


def test_cross_entropy_out_of_range_class_clamps_to_zero():
    """An out-of-range label is clamped to class 0 rather than raising.

    Documented, not endorsed: it makes a corrupted label train the model
    towards class 0 instead of failing loudly. The clamp is pinned so a change
    to it is a deliberate decision.
    """
    from core import losses

    logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    in_range, _ = losses.CrossEntropyLoss()(logits, np.array([0, 1]))
    out_of_range, grad = losses.CrossEntropyLoss()(logits, np.array([99, 1]))

    p = _softmax(logits)
    # Clamping class 99 to class 0 makes row 0 identical to an explicit target 0.
    assert abs(out_of_range - in_range) < LOG_TOL
    assert np.allclose(grad[0], (p[0] - np.array([1.0, 0.0, 0.0])) / 2)


def test_cross_entropy_stays_finite_on_large_logits():
    """exp(1000) overflows to inf, so the max-subtraction is load-bearing."""
    from core import losses

    logits = np.array([[1000.0, -1000.0, 0.0]])
    loss, grad = losses.CrossEntropyLoss()(logits, np.array([0]))

    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))
    # A saturated softmax: the confident class takes all the mass.
    assert abs(loss) < 1e-9, loss
    assert np.allclose(grad, np.zeros_like(logits), atol=1e-9)


def test_cross_entropy_single_sample_logits():
    from core import losses

    logits = np.array([1.0, 2.0, 3.0])
    loss, grad = losses.CrossEntropyLoss()(logits, np.array([2]))

    p = _softmax(logits)
    assert abs(loss - (-np.log(p[2]))) < LOG_TOL
    assert np.allclose(grad, p - np.array([0.0, 0.0, 1.0]))


def test_cross_entropy_matches_central_differences():
    """Gradient-checked against the loss it defines, per logit entry."""
    from core import losses

    loss_fn = losses.CrossEntropyLoss()
    logits = np.array([[1.5, -0.5, 0.25], [0.3, 2.1, -1.4]])
    target = np.array([0, 1])
    _loss, grad = loss_fn(logits, target)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            plus = logits.copy()
            plus[i, j] += 1e-6
            minus = logits.copy()
            minus[i, j] -= 1e-6
            numeric = central_diff(
                loss_fn(plus, target)[0],
                loss_fn(minus, target)[0],
            )
            assert abs(numeric - grad[i, j]) < 1e-6, (i, j, numeric, grad[i, j])


def test_cross_entropy_gradient_reaches_linear_weights():
    """End to end: the dL/dlogits contract has to survive a real backward.

    Uses ReLU with a bias that keeps every pre-activation positive, so the
    activation derivative is exactly 1 and grad_w is the plain outer product.
    (nn.Linear has no identity activation to select; see the module docstring.)
    """
    from core import losses, nn

    net = nn.Linear(
        3, 2, activation='relu', weights=np.zeros(6), bias=np.array([10.0, 10.0])
    )
    x = np.array([0.5, -0.25, 1.5])

    before = net.weights().copy()
    _loss, grad_out = losses.CrossEntropyLoss()(net(x), np.array([1]))
    net.backward(grad_out)

    grad_w = net.grad_weights().reshape(2, 3)
    # grad_w = x outer grad_out, so each entry is grad_out[i] * x[j].
    assert np.allclose(grad_w, np.outer(grad_out, x))
    moved = before - 0.1 * grad_w.reshape(-1)
    assert not np.allclose(before, moved), "weights should move"


def test_linear_cross_entropy_training_step_decreases_the_loss():
    """The supported pairing trains: Linear -> CrossEntropyLoss, no Softmax layer."""
    from core import losses, nn, optim

    # ReLU rather than the default activation, which is sigmoid: a saturating
    # sigmoid between the logits and the loss would cap how far the gradient can
    # push the target class up.
    #
    # The small positive bias is not decoration. With weights and bias all zero
    # the pre-activation sits exactly at 0, where ReLU's derivative is 0, so the
    # gradient is gated shut and the loss never moves. Seeding the biases above
    # zero puts the layer in the live region.
    net = nn.Linear(
        2, 2, activation='relu', weights=np.zeros(4), bias=np.array([0.1, 0.1])
    )
    loss_fn = losses.CrossEntropyLoss()
    opt = optim.SGD(list(net.parameters()), lr=0.5)

    x = np.array([1.0, -0.5])
    target = np.array([0])

    net.train()
    first = None
    for _ in range(20):
        loss, grad_out = loss_fn(net(x), target)
        net.backward(grad_out)
        if first is None:
            first = loss
        opt.step()
        opt.zero_grad()

    assert net(x)[0] > net(x)[1], "should learn to prefer the target class"
    assert loss < first, (first, loss)