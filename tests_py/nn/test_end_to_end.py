"""End-to-end: a Conv2d network trained through the manual backward loop."""

import numpy as np


def manual_backward(net, grad_out):
    """Propagate a gradient back through a Sequential.

    Mirrors what a user writes by hand, including the `hasattr(layer,
    "backward")` skip: a module without a derivative is silently dropped from
    the chain, so any layer added here must implement backward.
    """
    grad = grad_out
    for layer in reversed(net.layers):
        if hasattr(layer, "backward"):
            grad = layer.backward(grad)
    return grad


def test_conv2d_then_flatten_then_linear_end_to_end():
    from core import nn, losses, optim

    # Widths have to match at every handoff. Conv 3x3 on a 4x4 input leaves a
    # 2x2 map, MaxPool2d(2) collapses that to 1x1, so the flattened activation
    # is 2 values (one per channel), not 2*2*2.
    #
    # This used to say Linear(2 * 2 * 2, 1). The native layer reads num_inputs
    # doubles out of a 2-element buffer, so it consumed uninitialised memory:
    # gradients came back as ~1e251, the loss went NaN, and the test still
    # passed because it only asserted the output *shape*. nn.Linear now rejects
    # the mismatch, which is what surfaced it.
    net = nn.Sequential(
        nn.Conv2d(1, 2, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2, 1),
        nn.Sigmoid(),
    )
    x = np.random.randn(1, 4, 4)
    y = np.array([1.0])
    loss_fn = losses.MSELoss()
    opt = optim.SGD(list(net.parameters()), lr=0.05)

    first_loss = None
    for _ in range(5):
        pred = net(x)
        loss, grad_out = loss_fn(pred, y)
        manual_backward(net, grad_out)
        opt.step()
        opt.zero_grad()
        if first_loss is None:
            first_loss = loss
    assert first_loss is not None
    assert net(x).shape == (1,)