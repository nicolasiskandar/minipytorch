"""minipytorch demo: train an MLP on XOR, run a small CNN, and round-trip a model.

Run from the repo root:

    .venv/bin/python demo.py
"""

import tempfile

import numpy as np

from core import io, losses, nn, optim

XOR_INPUTS = [
    np.array([0.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 0.0]),
    np.array([1.0, 1.0]),
]
XOR_TARGETS = [np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([0.0])]


def demo_xor():
    print("=== 1. MLP on XOR ===")
    net = nn.Sequential(
        nn.Linear(2, 8, activation="relu"),
        nn.Linear(8, 1, activation="sigmoid"),
    )
    loss_fn = losses.MSELoss()
    opt = optim.SGD(list(net.parameters()), lr=0.3, momentum=0.9)

    def avg_loss():
        return float(np.mean([loss_fn(net(x), y)[0] for x, y in zip(XOR_INPUTS, XOR_TARGETS)]))

    print(f"loss before training: {avg_loss():.6f}")

    epochs = 1500
    for epoch in range(epochs):
        for x, y in zip(XOR_INPUTS, XOR_TARGETS):
            loss, grad_out = loss_fn(net(x), y)
            net.backward(grad_out)
            opt.step()
            opt.zero_grad()
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch + 1:4d}  loss: {avg_loss():.6f}")

    print("predictions:")
    for x, y in zip(XOR_INPUTS, XOR_TARGETS):
        pred = float(net(x)[0])
        print(f"  {x.astype(int)} -> {pred:.4f}  (target {int(y[0])}, rounded {round(pred)})")
    print()


def demo_cnn():
    print("=== 2. CNN (Conv2d -> ReLU -> MaxPool -> Flatten -> Linear) ===")
    net = nn.Sequential(
        nn.Conv2d(1, 2, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2, 1),
        nn.Sigmoid(),
    )
    x = np.random.default_rng(0).standard_normal((1, 4, 4))
    y = np.array([1.0])
    loss_fn = losses.MSELoss()
    opt = optim.SGD(list(net.parameters()), lr=0.05)

    out = net(x)
    print(f"output shape: {out.shape}")

    for step in range(50):
        loss, grad_out = loss_fn(net(x), y)
        net.backward(grad_out)
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            print(f"step {step:3d}  loss: {loss:.6f}")
    print(f"step  49  loss: {loss:.6f}")
    print()


def demo_save_load():
    print("=== 3. save / load round-trip ===")
    net = nn.Sequential(
        nn.Linear(2, 4, activation="tanh"),
        nn.Linear(4, 1, activation="sigmoid"),
    )
    x = np.array([0.5, -0.25])

    with tempfile.NamedTemporaryFile(suffix=".mpk") as f:
        io.save_model(net, f.name)
        restored = io.load_model(f.name)

    before, after = net(x), restored(x)
    print(f"original output: {before}")
    print(f"restored output: {after}")
    print(f"outputs match:   {np.allclose(before, after)}")


if __name__ == "__main__":
    demo_xor()
    demo_cnn()
    demo_save_load()
