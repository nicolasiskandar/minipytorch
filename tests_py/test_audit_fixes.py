"""Regression tests for the audit fixes.

Each test pins a specific bug that was confirmed and then fixed. The names
spell out the failure mode so a regression is diagnosable from the test name.
"""

import os
import tempfile

import numpy as np

from tests_py._helpers import central_diff


def test_linear_constructor_rejects_short_and_overlong_weights():
    from core import nn

    for bad in ([1.0, 2.0], np.ones(7), np.ones(0)):
        try:
            nn.Linear(2, 3, weights=bad)
            raise AssertionError(f"expected ValueError for weights of size {np.size(bad)}")
        except ValueError:
            pass


def test_linear_constructor_rejects_short_and_overlong_bias():
    from core import nn

    for bad in ([1.0], np.ones(5)):
        try:
            nn.Linear(2, 3, bias=bad)
            raise AssertionError(f"expected ValueError for bias of size {np.size(bad)}")
        except ValueError:
            pass


def test_linear_rejects_negative_feature_counts():
    from core import nn

    for a, b in ((-1, 3), (2, -1)):
        try:
            nn.Linear(a, b)
            raise AssertionError(f"expected ValueError for Linear({a}, {b})")
        except ValueError:
            pass


def test_conv2d_backward_rejects_wrong_sized_gradients():
    from core import nn

    conv = nn.Conv2d(1, 1, 2)
    conv(np.ones((1, 3, 3)))  # output is (1, 2, 2) -> 4 values
    for bad_size in (0, 7, 100):
        try:
            conv.backward(np.zeros(bad_size))
            raise AssertionError(f"expected ValueError for grad of size {bad_size}")
        except ValueError:
            pass


def test_maxpool2d_backward_rejects_wrong_sized_gradients():
    from core import nn

    pool = nn.MaxPool2d(2)
    pool(np.ones((1, 4, 4)))  # output 2x2 -> 4 values
    try:
        pool.backward(np.zeros(3))
        raise AssertionError("expected ValueError for 3-element grad")
    except ValueError:
        pass


def test_pool_and_conv_reject_zero_stride():
    from core import nn

    for make in (
        lambda: nn.MaxPool2d(2, stride=0),
        lambda: nn.AvgPool2d(2, stride=0),
        lambda: nn.Conv2d(1, 1, 2, stride=0),
    ):
        try:
            make()
            raise AssertionError("expected ValueError for stride=0")
        except ValueError:
            pass


def test_non_square_strides_rejected_rather_than_truncated():
    """The native kernels take a single stride/padding for both dims, so a
    non-square stride used to be silently truncated to component [0]. We now
    refuse the configuration instead of computing something wrong."""
    from core import nn

    for make in (
        lambda: nn.MaxPool2d(2, stride=(1, 2)),
        lambda: nn.AvgPool2d(2, padding=(0, 1)),
        lambda: nn.Conv2d(1, 1, 2, stride=(1, 2)),
    ):
        try:
            make()
            raise AssertionError("expected ValueError for non-square stride/padding")
        except ValueError:
            pass


def test_conv_maxpool_flatten_linear_backward_flows_top_to_bottom():
    """The Conv -> ReLU -> MaxPool -> Flatten -> Linear chain now backprops
    all the way to the input instead of crashing on the shape mismatch at the
    ReLU/Flatten handoff."""
    from core import nn

    net = nn.Sequential(
        nn.Conv2d(1, 2, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2, 1),
    )
    x = np.random.randn(1, 4, 4)
    out = net(x)
    grad = net.backward(np.ones_like(out))
    assert grad.shape == x.shape


def test_elementwise_backward_rejects_wrong_sized_flat_gradient():
    from core import nn

    relu = nn.ReLU()
    relu(np.ones((2, 2)))
    try:
        relu.backward(np.ones(3))
        raise AssertionError("expected ValueError for 3-element grad on 4-element layer")
    except ValueError:
        pass


def test_mse_loss_is_mean_reduced_and_gradient_consistent():
    from core import losses

    loss_fn = losses.MSELoss()
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.zeros(4)
    loss, grad = loss_fn(pred, target)

    assert abs(loss - np.mean(pred**2)) < 1e-10
    assert np.allclose(grad, 2.0 * pred / 4.0)

    for i in range(pred.size):
        plus = pred.copy()
        plus[i] += 1e-6
        minus = pred.copy()
        minus[i] -= 1e-6
        numeric = central_diff(loss_fn(plus, target)[0], loss_fn(minus, target)[0])
        assert abs(numeric - grad[i]) < 1e-6, (i, numeric, grad[i])


def test_shared_layer_in_sequential_applies_gradient_once_with_momentum():
    """A module listed twice (Sequential(shared, ReLU, shared)) yields 4
    parameter handles wrapping 2 storages. SGD with momentum used to apply the
    gradient twice; it must apply it exactly once."""
    from core import nn, optim

    shared = nn.Linear(1, 1, activation='relu', weights=np.array([1.0]), bias=np.array([1.0]))
    net = nn.Sequential(shared, nn.ReLU(), shared)
    net(np.array([3.0]))  # pre-activations are > 0, so relu is fully live

    net.backward(np.array([1.0]))
    grad_w = shared.grad_weights()

    before = shared.weight.copy()
    optim.SGD(list(net.parameters()), lr=1.0, momentum=0.9).step()
    moved = np.abs(shared.weight - before)
    # v = g (first step), so w -= w + lr*g: moved == |g| when applied once,
    # == 2|g| when the duplicate handle double-applies.
    assert np.allclose(moved, np.abs(grad_w), atol=1e-12)


def test_shared_layer_in_sequential_applies_gradient_once_with_adam():
    from core import nn, optim

    shared = nn.Linear(1, 1, activation='relu', weights=np.array([1.0]), bias=np.array([1.0]))
    net = nn.Sequential(shared, nn.ReLU(), shared)
    net(np.array([3.0]))
    net.backward(np.array([1.0]))

    before = shared.weight.copy()
    optim.Adam(list(net.parameters()), lr=1.0).step()
    moved = np.abs(shared.weight - before)
    grad = np.abs(np.asarray(shared.grad_weights()))
    expected = grad / (grad + 1e-8)
    assert np.allclose(moved, expected, atol=1e-6)


def test_conv2d_parameter_handles_are_stable_across_calls():
    """optim.Parameter keeps its momentum buffer; identity handles let a
    rebuilt optimizer keep its state instead of silently resetting it."""
    from core import nn

    conv = nn.Conv2d(1, 2, 2)
    first = list(conv.parameters())
    second = list(conv.parameters())
    assert len(first) == 2
    assert all(a is b for a, b in zip(first, second))


def test_softmax_dim_survives_roundtrip():
    import numpy as np

    from core import io, nn

    net = nn.Sequential(
        nn.Linear(4, 4, weights=np.eye(4).reshape(-1), bias=np.zeros(4)),
        nn.Softmax(dim=0),
    )
    x = np.arange(4.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "softmax.mpk")
        io.save_model(net, path)
        reloaded = io.load_model(path)
    assert reloaded.layers[-1].dim == 0
    assert np.allclose(reloaded(x), net(x))


def test_train_eval_mode_survives_roundtrip():
    from core import io, nn

    net = nn.Sequential(nn.Linear(2, 2), nn.Dropout(p=0.5))
    net.eval()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mode.mpk")
        io.save_model(net, path)
        reloaded = io.load_model(path)

    assert not reloaded.training()
    assert not reloaded.layers[-1].training()


def test_nested_sequential_can_be_saved():
    from core import io, nn

    net = nn.Sequential(
        nn.Sequential(nn.ReLU(), nn.Sigmoid()),
        nn.Linear(4, 1, weights=np.ones(4), bias=np.zeros(1)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested.mpk")
        io.save_model(net, path)
        reloaded = io.load_model(path)
    assert [type(l).__name__ for l in reloaded.layers] == ["ReLU", "Sigmoid", "Linear"]


def test_linear_load_rejects_wrong_weight_count():
    from core import io
    import tempfile

    text = "\n".join(
        [
            "minipytorch-v2",
            "1",
            "linear 2 3 sigmoid",
            "2",  # count does not match out_features=3
            "0.0 1.0 2.0",
            "0.0 1.0 2.0",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.mpk")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        try:
            io.load_model(path)
            raise AssertionError("expected ValueError for count mismatch")
        except ValueError:
            pass


def test_malformed_files_raise_value_error_not_index_error():
    from core import io
    import tempfile

    bad_texts = [
        "minipytorch-v2\n",  # no layer count
        "minipytorch-v2\n-3\n",  # negative layer count
        "minipytorch-v2\n1\nlinear 2 -3 sigmoid\n",  # negative out_features
        "minipytorch-v2\n1\nbogus_layer\n",  # unknown tag
    ]
    for i, text in enumerate(bad_texts):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"bad{i}.mpk")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            try:
                io.load_model(path)
                raise AssertionError(f"expected ValueError for: {text!r}")
            except ValueError:
                pass


def test_activation_output_mutation_does_not_corrupt_backward():
    from core import nn

    sig = nn.Sigmoid()
    out = sig(np.array([0.0, 0.0]))
    # Mutating the returned array must not touch the cached derivative.
    out[:] = 99.0
    grad = sig.backward(np.ones(2))
    # d/dx sigmoid(0) = 0.25.
    assert np.allclose(grad, 0.25)


def test_flatten_returns_a_private_copy():
    from core import nn

    x = np.arange(6.0).reshape(2, 3)
    flat = nn.Flatten()(x)
    flat[0] = 123.0
    assert x[0, 0] != 123.0


def test_dropout_backward_before_forward_in_train_raises():
    from core import nn

    layer = nn.Dropout(p=0.5)
    layer.train()
    try:
        layer.backward(np.ones(4))
        raise AssertionError("expected RuntimeError for training backward-without-forward")
    except RuntimeError:
        pass


def test_leaky_relu_inplace_is_rejected():
    from core import nn

    try:
        nn.LeakyReLU(inplace=True)
        raise AssertionError("expected NotImplementedError for inplace=True")
    except NotImplementedError:
        pass