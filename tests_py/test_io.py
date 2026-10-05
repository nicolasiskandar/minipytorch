"""Serialization: the minipytorch-v2 format, its round-trips, and its rejects."""

import numpy as np

from tests_py._helpers import load_model_error, model_path


def test_save_load_roundtrip():
    from core import nn, io

    l1 = nn.Linear(2, 3, activation='tanh')
    l2 = nn.Linear(3, 1, activation='sigmoid')
    seq = nn.Sequential(l1, l2)
    x = np.array([0.5, -0.3])
    out1 = seq(x)

    with model_path() as f:
        io.save_model(seq, f)
        seq2 = io.load_model(f)

    out2 = seq2(x)
    assert np.allclose(out1, out2)


def test_save_load_roundtrip_all_activations():
    from core import nn, io

    x = np.array([0.5, -0.3])
    for act in ['sigmoid', 'tanh', 'relu', 'relu6', 'leaky_relu']:
        l1 = nn.Linear(2, 3, activation=act)
        l2 = nn.Linear(3, 1, activation=act)
        seq = nn.Sequential(l1, l2)
        expected = seq(x)

        with model_path() as f:
            io.save_model(seq, f)
            reloaded = io.load_model(f)

        assert int(reloaded.layers[0].activation_kind) == int(l1.activation_kind), act
        assert int(reloaded.layers[1].activation_kind) == int(l2.activation_kind), act
        assert np.allclose(reloaded(x), expected), act


def test_save_model_writes_activation_name_verbatim():
    from core import nn, io

    for act in ['relu6', 'leaky_relu']:
        seq = nn.Sequential(nn.Linear(2, 2, activation=act))
        with model_path() as f:
            io.save_model(seq, f)
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()
        assert act in text, act
        assert "sigmoid" not in text, act


def test_load_model_rejects_unknown_activation():
    with model_path() as f:
        with open(f, "w", encoding="utf-8") as out:
            out.write("minipytorch-v2\n1\nlinear 2 1 gelu\n1\n0.5 1.0 2.0\n")
        message = load_model_error(f)
    assert "unknown activation" in message, message
    assert "gelu" in message, message


def test_load_model_rejects_malformed_bias():
    with model_path() as f:
        with open(f, "w", encoding="utf-8") as out:
            out.write("minipytorch-v2\n1\nlinear 2 1 relu\n1\nnotanumber 1.0 2.0\n")
        message = load_model_error(f)
    assert "could not convert string to float" in message, message


def test_load_model_rejects_untagged_file():
    """The retired Linear-only positional layout must not parse."""
    with model_path() as f:
        with open(f, "w", encoding="utf-8") as out:
            out.write("1\n1\nrelu 0.5 1.0 2.0\n")
        message = load_model_error(f)
    assert "not a minipytorch-v2 model" in message, message


def test_save_load_roundtrip_conv_pool_network():
    from core import nn, io

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

    with model_path() as path:
        io.save_model(net, path)
        reloaded = io.load_model(path)

    assert len(reloaded.layers) == len(net.layers)
    assert np.allclose(reloaded(x), expected)


def test_save_load_preserves_conv_hyperparameters():
    from core import nn, io

    net = nn.Sequential(
        nn.Conv2d(2, 3, 3, stride=2, padding=1), nn.AvgPool2d(2, stride=1)
    )
    x = np.random.RandomState(1).randn(2, 6, 6)
    expected = net(x)

    with model_path() as path:
        io.save_model(net, path)
        reloaded = io.load_model(path)

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

    net = nn.Sequential(nn.Linear(2, 2), nn.LeakyReLU(negative_slope=0.25))
    x = np.array([1.0, -2.0])
    expected = net(x)

    with model_path() as path:
        io.save_model(net, path)
        reloaded = io.load_model(path)

    assert reloaded.layers[1].negative_slope == 0.25
    assert np.allclose(reloaded(x), expected)


def test_load_model_rejects_unknown_layer_tag():
    from core import io

    with model_path() as path:
        with open(path, "w", encoding="utf-8") as out:
            out.write("minipytorch-v2\n1\nbanana 0\n")
        try:
            io.load_model(path)
            raise AssertionError("expected ValueError for unknown layer tag")
        except ValueError:
            pass


def test_load_model_rejects_inconsistent_conv_weight_count():
    from core import io

    with model_path() as path:
        with open(path, "w", encoding="utf-8") as out:
            out.write("minipytorch-v2\n1\nconv2d 1 2 3 3 1 0 1\n17 2\n")
            out.write(" ".join(["0.0"] * 17) + "\n")
            out.write("0.0 0.0\n")
        try:
            io.load_model(path)
            raise AssertionError("expected ValueError for wrong weight count")
        except ValueError:
            pass