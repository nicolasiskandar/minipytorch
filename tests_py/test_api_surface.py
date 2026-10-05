CORE_ALL = [
    "ActivationKind",
    "Adam",
    "AvgPool2d",
    "BCELoss",
    "Conv2d",
    "CrossEntropyLoss",
    "FastLayer",
    "Flatten",
    "LeakyReLU",
    "Linear",
    "MSELoss",
    "MaxPool2d",
    "ReLU",
    "ReLU6",
    "SGD",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "activations",
    "bce_loss",
    "io",
    "leaky_relu_derivative_from_output_f64",
    "leaky_relu_f64",
    "load_model",
    "losses",
    "mse_loss",
    "nn",
    "optim",
    "relu6_deriv_from_output_f64",
    "relu6_f64",
    "relu_deriv_from_output_f64",
    "relu_f64",
    "save_model",
    "sigmoid_deriv_from_output_f64",
    "sigmoid_f64",
    "tanh_deriv_from_output_f64",
    "tanh_f64",
]

CORE_NOT_IN_ALL = [
    "Module",
    "Parameter",
    "leaky_relu",
    "leaky_relu_deriv",
    "relu",
    "relu6",
    "relu6_deriv",
    "relu_deriv",
    "sigmoid",
    "sigmoid_deriv",
    "softmax",
    "tanh",
    "tanh_deriv",
]

NN_ALL = [
    "AvgPool2d",
    "Conv2d",
    "Flatten",
    "LeakyReLU",
    "Linear",
    "MaxPool2d",
    "Module",
    "Parameter",
    "ReLU",
    "ReLU6",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
]

NN_NOT_IN_ALL = [
    "ActivationKind",
    "FastLayer",
    "avgpool2d_backward",
    "avgpool2d_forward",
    "conv2d_backward_input",
    "conv2d_backward_weight",
    "conv2d_forward",
    "maxpool2d_backward",
    "maxpool2d_forward",
]

ACTIVATION_KIND_MEMBERS = ["SIGMOID", "TANH", "RELU", "RELU6", "LEAKYRELU", "LEAKY_RELU"]

ACTIVATIONS_ALL = [
    "ActivationKind",
    "leaky_relu",
    "leaky_relu_deriv",
    "relu",
    "relu6",
    "relu6_deriv",
    "relu_deriv",
    "sigmoid",
    "sigmoid_deriv",
    "softmax",
    "tanh",
    "tanh_deriv",
]

OPTIM_ALL = ["Adam", "SGD"]
LOSSES_ALL = ["BCELoss", "CrossEntropyLoss", "MSELoss", "bce_loss", "mse_loss"]
IO_ALL = ["load_model", "save_model"]

STATELESS_LAYER_TAGS = [
    "avgpool2d",
    "flatten",
    "leaky_relu",
    "maxpool2d",
    "relu",
    "relu6",
    "sigmoid",
    "softmax",
    "tanh",
]

ACTIVATION_NAMES = ["leaky_relu", "relu", "relu6", "sigmoid", "tanh"]

EXPECTED_CLASS = {
    "avgpool2d": "AvgPool2d",
    "flatten": "Flatten",
    "leaky_relu": "LeakyReLU",
    "maxpool2d": "MaxPool2d",
    "relu": "ReLU",
    "relu6": "ReLU6",
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "tanh": "Tanh",
}

ARG_NAMES = {
    "avgpool2d": ("kernel_size", "stride", "padding"),
    "flatten": ("start_dim",),
    "leaky_relu": ("negative_slope",),
    "maxpool2d": ("kernel_size", "stride", "padding"),
    "relu": (),
    "relu6": (),
    "sigmoid": (),
    "softmax": (),
    "tanh": (),
}

MODULE_SUBCLASSES = [
    "AvgPool2d",
    "Conv2d",
    "Flatten",
    "LeakyReLU",
    "Linear",
    "MaxPool2d",
    "ReLU",
    "ReLU6",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
]


def test_core_all_resolves():
    from core import activations, io, losses, nn, optim

    import core

    missing = [name for name in CORE_ALL if not hasattr(core, name)]
    assert missing == [], f"core.__all__ names not reachable: {missing}"
    assert sorted(CORE_ALL) == sorted(core.__all__), "core.__all__ drifted"
    for mod in (activations, io, losses, nn, optim):
        assert mod.__name__.startswith("core.")


def test_core_names_outside_all_are_exactly_as_expected():
    import core

    extra = [
        name
        for name in dir(core)
        if not name.startswith("_") and name not in CORE_ALL
    ]
    assert sorted(extra) == sorted(CORE_NOT_IN_ALL), (
        "names reachable as core.<name> but absent from core.__all__ changed; "
        "a split submodule may be leaking into the package namespace"
    )


def test_nn_all_resolves():
    from core import nn

    missing = [name for name in NN_ALL if not hasattr(nn, name)]
    assert missing == [], f"nn.__all__ names not reachable: {missing}"
    assert sorted(NN_ALL) == sorted(nn.__all__), "nn.__all__ drifted"


def test_nn_names_outside_all_are_exactly_as_expected():
    import types

    from core import nn

    extra = [
        name
        for name in dir(nn)
        if not name.startswith("_") and name not in NN_ALL
    ]
    # Incidental imports (`numpy as np`) live in the leaf modules after a
    # split, so module objects are not part of the contracted surface.
    extra = [name for name in extra if not isinstance(getattr(nn, name), types.ModuleType)]
    assert sorted(extra) == sorted(NN_NOT_IN_ALL), (
        "nn exposes names outside __all__; expected exactly "
        f"{NN_NOT_IN_ALL}, got {sorted(extra)}"
    )


def test_split_leaf_modules_do_not_shadow_real_modules():
    """The hazard: `from .nn import *` picking up a submodule instead of a class.

    Once nn.py becomes a package it has submodules named `activations`,
    `linear`, `conv`, ... A sloppy aggregator `__all__` would bind
    `core.nn.activations` over the real `core.activations`, silently replacing
    the functional kernels with a module object.
    """
    import types

    import core
    import core.activations as activations
    import core.io as io
    import core.losses as losses
    import core.nn as nn
    import core.optim as optim

    expected = {
        "activations": activations,
        "io": io,
        "losses": losses,
        "nn": nn,
        "optim": optim,
    }
    for name, module in expected.items():
        assert getattr(core, name) is module, (
            f"core.{name} is {getattr(core, name)!r}, not the real module; a "
            "split submodule has shadowed it"
        )

    modules = {
        name
        for name in dir(core)
        if isinstance(getattr(core, name), types.ModuleType)
        and not name.startswith("_")
    }
    assert modules == set(expected), (
        f"core exposes modules {sorted(modules)}, expected {sorted(expected)}"
    )

    # A submodule binding on the package itself is normal and expected; what
    # must not happen is one of them escaping into core's namespace.
    contracted = set(CORE_ALL) | set(CORE_NOT_IN_ALL)
    for name in set(dir(nn)) | set(dir(io)):
        if name.startswith("_") or name in contracted:
            continue
        assert not hasattr(core, name), (
            f"core.{name} leaked from a split submodule"
        )


def test_activation_kind_is_reachable_through_nn():
    from core import nn
    from core._minipytorch import ActivationKind

    assert nn.ActivationKind is ActivationKind
    values = {getattr(ActivationKind, n).value for n in ACTIVATION_KIND_MEMBERS}
    assert values == {0, 1, 2, 3, 4}
    # Value 4 is exported twice, as LEAKYRELU and LEAKY_RELU. They compare and
    # hash by value, but are distinct objects, and the constructor returns a
    # third. So never rely on identity or on round-tripping through the name.
    assert ActivationKind.LEAKYRELU == ActivationKind.LEAKY_RELU
    assert ActivationKind.LEAKYRELU is not ActivationKind.LEAKY_RELU
    assert ActivationKind(4) is not ActivationKind.LEAKYRELU
    assert ActivationKind(4) == ActivationKind.LEAKYRELU


def test_aggregators_reuse_the_same_class_objects():
    import core
    from core import io, nn

    for name in NN_ALL + ["Module", "Parameter"]:
        assert getattr(core, name) is getattr(nn, name), (
            f"core.{name} and nn.{name} are different objects; a class is "
            "defined twice, so isinstance checks in core.io will not match"
        )
    assert core.save_model is io.save_model
    assert core.load_model is io.load_model


def test_activation_kind_is_shared_between_activations_and_io():
    import core
    from core import activations, nn

    assert activations.ActivationKind is nn.ActivationKind
    assert core.ActivationKind is activations.ActivationKind


def test_activations_optim_losses_io_exports():
    from core import activations, io, losses, optim

    assert sorted(activations.__all__) == sorted(ACTIVATIONS_ALL)
    assert sorted(optim.__all__) == sorted(OPTIM_ALL)
    assert sorted(losses.__all__) == sorted(LOSSES_ALL)
    assert sorted(io.__all__) == sorted(IO_ALL)
    for module in (activations, optim, losses, io):
        missing = [name for name in module.__all__ if not hasattr(module, name)]
        assert missing == [], f"{module.__name__}: {missing}"


def test_module_subclass_relationships_hold():
    from core import nn

    for name in MODULE_SUBCLASSES:
        cls = getattr(nn, name)
        assert issubclass(cls, nn.Module), f"{name} no longer subclasses Module"
    for name in ("ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ReLU6"):
        cls = getattr(nn, name)
        assert cls.__bases__ == (nn._ElementwiseActivation,), (
            f"{name} must subclass _ElementwiseActivation exactly; its backward "
            "gates the gradient chain"
        )


def test_activation_modules_implement_backward():
    from core import nn

    for name in ("ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ReLU6"):
        cls = getattr(nn, name)
        assert callable(getattr(cls, "backward", None)), (
            f"nn.{name} has no backward; a Sequential backward loop silently "
            "skips it and passes gradient through dead units"
        )


def test_softmax_backward_is_still_missing():
    """Documents a known gap, not an endorsement.

    nn.Softmax implements forward only. A manual backward loop skips layers
    without `backward`, so a Sequential ending in Softmax trains against a
    gradient that never passed the softmax Jacobian. Deliberately asserted here
    so that whoever implements it gets told to fold Softmax into
    test_activation_modules_implement_backward instead of quietly deleting this.
    """
    from core import nn

    assert not hasattr(nn.Softmax, "backward")


def test_flatten_has_no_backward_by_design():
    from core import nn

    # Its gradient is the identity reshape, so the caller's gradient flows
    # through untouched. Documented as intentional, not a missing derivative.
    assert not hasattr(nn.Flatten, "backward")


def test_sequential_is_fully_duck_typed():
    from core import nn
    import inspect

    src = inspect.getsource(nn.Sequential)
    for coupling in ("nn.Linear", "Conv2d", "Flatten", "ReLU"):
        assert coupling not in src, (
            f"Sequential must not reference {coupling} directly; it dispatches "
            "on hasattr so any Module can be nested"
        )


def test_io_format_constants_survive():
    from core import io

    assert io._FORMAT_TAG == "minipytorch-v2"
    assert sorted(io._ACTIVATION_NAMES.values()) == sorted(ACTIVATION_NAMES)
    assert sorted(io._REVERSE_ACTIVATION_NAMES) == sorted(ACTIVATION_NAMES)
    assert io._ACTIVATION_NAMES == {
        value: name for name, value in io._REVERSE_ACTIVATION_NAMES.items()
    }


def test_io_activation_names_cover_every_activation_kind():
    from core import io
    from core.activations import ActivationKind

    for member in ACTIVATION_KIND_MEMBERS:
        kind = getattr(ActivationKind, member)
        assert int(kind) in io._ACTIVATION_NAMES, (
            f"ActivationKind.{member} is missing from io._ACTIVATION_NAMES; "
            "save_model would relabel the layer instead of raising"
        )


def test_io_activation_names_round_trip_through_text():
    from core import io

    for kind_int, name in io._ACTIVATION_NAMES.items():
        parsed = io._REVERSE_ACTIVATION_NAMES[name]
        assert int(parsed) == kind_int, f"{name} round-trip changed kind"
        assert io._get_activation_name(parsed) == name


def test_io_stateless_layer_registry_is_intact():
    """The factory must be the class itself, for all nine tags.

    ``_save_tagged`` finds a layer's tag with ``type(layer) is factory``, so a
    wrapper such as ``lambda: Softmax()`` makes the layer unsaveable while the
    reader -- which calls ``factory()`` -- keeps working. That asymmetry is
    invisible until you try to save a network containing the layer.
    """
    from core import io, nn

    registry = io._STATELESS_LAYERS
    assert sorted(registry) == sorted(STATELESS_LAYER_TAGS)
    for tag, class_name in EXPECTED_CLASS.items():
        factory, arg_names = registry[tag]
        expected = getattr(nn, class_name)
        assert factory is expected, (
            f"registry[{tag!r}] factory is {factory!r}, not nn.{class_name}; "
            "matching is by exact type(), so the factory must be the class"
        )
        assert arg_names == ARG_NAMES[tag], (
            f"registry[{tag!r}] argument names changed, so the on-disk "
            f"hyperparameter order changed: {arg_names}"
        )


def test_every_registered_stateless_layer_round_trips():
    """A registered tag that cannot be saved is a hole in the format."""
    import os
    import tempfile

    import numpy as np
    from core import io, nn

    builders = {
        "relu": lambda: nn.ReLU(),
        "sigmoid": lambda: nn.Sigmoid(),
        "tanh": lambda: nn.Tanh(),
        "relu6": lambda: nn.ReLU6(),
        "softmax": lambda: nn.Softmax(),
        "flatten": lambda: nn.Flatten(),
        "maxpool2d": lambda: nn.MaxPool2d(2),
        "avgpool2d": lambda: nn.AvgPool2d(2),
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.25),
    }
    assert sorted(builders) == sorted(STATELESS_LAYER_TAGS)

    with tempfile.TemporaryDirectory() as tmp:
        for tag, make in builders.items():
            # A leading Linear forces the tagged format, which is the only one
            # that carries layer tags.
            model = nn.Sequential(
                nn.Linear(2, 2, weights=np.ones(4), bias=np.zeros(2)), make()
            )
            path = os.path.join(tmp, f"{tag}.mpk")
            io.save_model(model, path)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            assert text.splitlines()[0] == io._FORMAT_TAG
            assert f"\n{tag}" in text or text.endswith(f"{tag}"), (
                f"{tag} layer was not written to {path}"
            )
            reloaded = io.load_model(path)
            assert type(reloaded.layers[-1]) is type(make()), (
                f"{tag} came back as {type(reloaded.layers[-1]).__name__}"
            )


def test_leaky_relu_slope_survives_the_text_format():
    """0.25 must not be routed through int(), which truncates it to 0."""
    import os
    import tempfile

    import numpy as np
    from core import io, nn

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "slope.mpk")
        io.save_model(
            nn.Sequential(
                nn.Linear(2, 2, weights=np.ones(4), bias=np.zeros(2)),
                nn.LeakyReLU(negative_slope=0.25),
            ),
            path,
        )
        reloaded = io.load_model(path)
        assert reloaded.layers[-1].negative_slope == 0.25, (
            f"slope came back as {reloaded.layers[-1].negative_slope}"
        )


def test_every_save_is_tagged_including_linear_only():
    """One format means the tag is on every file, Linear-only or not."""
    import os
    import tempfile

    import numpy as np
    from core import io, nn

    linear_only = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "linear_only.mpk")
        io.save_model(linear_only, path)
        with open(path, encoding="utf-8") as fh:
            assert fh.readline().strip() == io._FORMAT_TAG
        reloaded = io.load_model(path)
        assert [type(layer) for layer in reloaded.layers] == [nn.Linear, nn.Linear]

    with_conv = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=2, weights=np.ones(4), bias=np.zeros(1)),
        nn.Flatten(),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tagged.mpk")
        io.save_model(with_conv, path)
        with open(path, encoding="utf-8") as fh:
            assert fh.readline().strip() == io._FORMAT_TAG
        assert len(io.load_model(path).layers) == 2


def test_linear_carries_activation_kind_and_leaky_module_carries_slope():
    import numpy as np
    from core import nn

    leaky = nn.Linear(2, 2, activation="leaky_relu", weights=np.ones(4), bias=np.zeros(2))
    assert isinstance(leaky.activation_kind, nn.ActivationKind)

    module = nn.LeakyReLU(negative_slope=0.25)
    assert module.negative_slope == 0.25


def test_native_conv_and_pool_symbols_are_reachable_through_nn():
    import core
    from core import nn
    from core import _minipytorch as native

    for name in NN_NOT_IN_ALL:
        if name in ("ActivationKind", "FastLayer"):
            continue
        assert getattr(nn, name) is getattr(native, name), (
            f"nn.{name} is not the native symbol; it was re-wrapped"
        )
        assert getattr(core, name, None) is None or name in CORE_ALL


def test_numpy_is_not_leaked_into_exports():
    from core import nn

    assert "np" not in nn.__all__
    assert "np" not in NN_NOT_IN_ALL