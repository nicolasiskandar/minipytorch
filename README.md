# MiniPyTorch

A minimal PyTorch-like neural network library: a numpy frontend backed by
C++17 and hand-written x86-64 assembly kernels, exposed to Python through
pybind11.

Models are built from `Module` layers with explicit `forward`/`backward`
passes, there is no autograd engine, so backprop is driven layer by layer.
The hot numerical paths (dot products, activations, loss gradients, parameter
updates) run in assembly; C++ owns validation, object lifetime, and
orchestration.

## Features

- **Layers**: `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Flatten`,
  `Dropout`, activation modules (`ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`,
  `ReLU6`, `Softmax`), and `Sequential` containers
- **Losses** return `(loss, grad)` from a single call: `MSELoss`, `BCELoss`,
  `CrossEntropyLoss` (softmax folded in, pass raw logits)
- **Optimizers**: `SGD` (with momentum) and `Adam`
- **Serialization**: `save_model` / `load_model` using the tagged
  `minipytorch-v2` text format
- **Training modes**: `train()` / `eval()` recursion (matters for `Dropout`),
  `parameters()` / `named_parameters()` / `zero_grad()`
- **Native backend**: x86-64 assembly kernels for the hot paths, C++ for
  conv/pool and layer orchestration, pybind11 bindings

## Quick start

Requirements: `uv`, a C++17 compiler (`g++`), `make`.

```sh
./run.sh setup   # create .venv and install the package (builds the extension)
./run.sh demo    # run the end-to-end demo
```

`setup` runs `uv venv .venv && uv pip install -p .venv/bin/python -e .`:
scikit-build-core compiles the C++/assembly sources in an isolated build
environment and installs the `_minipytorch` extension into the package.
After that, `./run.sh test-py` runs the Python suite.

`demo.py` trains an MLP on XOR, runs a small CNN, and round-trips a model
through `save_model`/`load_model`. The core loop looks like this:

```python
import numpy as np
from core import nn, losses, optim

net = nn.Sequential(
    nn.Linear(2, 8, activation="relu"),
    nn.Linear(8, 1, activation="sigmoid"),
)
loss_fn = losses.MSELoss()                          # returns (loss, grad)
opt = optim.SGD(list(net.parameters()), lr=0.3, momentum=0.9)

x, y = np.array([0.0, 1.0]), np.array([1.0])        # one sample at a time
for _ in range(1500):
    loss, grad_out = loss_fn(net(x), y)
    net.backward(grad_out)
    opt.step()
    opt.zero_grad()                                 # grads accumulate otherwise

print(net(np.array([0.0, 1.0])))                    # -> [0.997...]
```

### Building the extension manually

To rebuild the extension outside of `pip`/`uv` (e.g. after changing a kernel),
use the CMake flow directly:

```sh
uv pip install -p .venv/bin/python cmake pybind11   # build deps, if missing
./run.sh build-ext
```

`build-ext` is equivalent to:

```sh
.venv/bin/cmake -S . -B build/ext \
  -Dpybind11_DIR=$(.venv/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())") \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build build/ext -j
.venv/bin/cmake --install build/ext --prefix "$PWD"   # installs into ./core/
```

## Running tests

```sh
./run.sh test-py    # 159 Python tests (tests_py/)
./run.sh test       # 177 C++ tests (backend_cxx/tests/)
./run.sh test-all   # both
```

The Python suite is run by an embedded runner (plain `assert`-style
`test_*` functions); pytest is not required.

## C++ experiments

```sh
./run.sh build      # build the C++ test/experiment binaries
./run.sh xor        # XOR experiment
./run.sh circle     # circle classification experiment
./run.sh bench      # FastLayer vs Layer benchmark
```

## Project structure

```
backend_cxx/         C++17 + assembly backend
  include/           Public headers (fastLayer, conv, pool, activations, losses)
  src/               C++ implementations (one module per subsystem)
  asm/kernels/       x86-64 scalar assembly numerical kernels
  tests/             C++ test suites and runner
  experiments/       XOR, circle classification, benchmark
  Makefile           Builds nn_test, nn_xor, nn_circle, nn_bench
core/                Python package (the public API)
  nn/                Module, Linear, Conv2d, pools, Dropout, activations, ...
  io/                save_model / load_model (minipytorch-v2 format)
  losses.py optim.py activations.py
src_py/bindings/     pybind11 translation units (one per backend subsystem)
tests_py/            Python test suite
CMakeLists.txt       Builds the _minipytorch extension
pyproject.toml       Packaging (scikit-build-core build backend)
run.sh               Task runner (setup, demo, test, build-ext, xor, circle, bench, build)
demo.py              End-to-end demo
```

## Python API

Import from the top-level `core` package: `from core import nn, losses, optim, io`.

### Layers

| Layer | Signature | Notes |
|-------|-----------|-------|
| `Linear` | `Linear(in_features, out_features, activation=None, weights=None, bias=None)` | 1-D input of size `in_features`. **Default activation is sigmoid** (`'relu'`, `'tanh'`, `'sigmoid'`, `'relu6'`, `'leaky_relu'`) |
| `Conv2d` | `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)` | Input `(C, H, W)`; square stride/padding only |
| `MaxPool2d` | `MaxPool2d(kernel_size, stride=None, padding=0)` | `stride` defaults to `kernel_size` |
| `AvgPool2d` | `AvgPool2d(kernel_size, stride=None, padding=0)` | Same constraints as `MaxPool2d` |
| `Flatten` | `Flatten(start_dim=1, end_dim=-1)` | e.g. `(C, H, W) -> (C, H*W)` |
| `Dropout` | `Dropout(p=0.5)` | Inverted scaling; active only in `train()` mode |
| `ReLU` / `Sigmoid` / `Tanh` / `ReLU6` | `()` | Stateless elementwise activations |
| `LeakyReLU` | `LeakyReLU(negative_slope=0.01)` | `inplace=True` is not supported |
| `Softmax` | `Softmax(dim=None)` | |
| `Sequential` | `Sequential(*layers)` | Chains `forward`, reverses for `backward` |

### Losses

All losses are callables returning `(loss: float, grad: np.ndarray)`:

```python
loss, grad_out = losses.MSELoss()(pred, target)
loss, grad_out = losses.BCELoss()(pred, target)           # pred in [0, 1]
loss, grad_out = losses.CrossEntropyLoss()(logits, target) # target = class index
```

`CrossEntropyLoss` folds in softmax: pair it with raw logits (no `Softmax`
layer), and pass `np.array([class_index])` as the target.

### Optimizers

```python
opt = optim.SGD(list(net.parameters()), lr=0.05, momentum=0.9)
opt = optim.Adam(list(net.parameters()), lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0)
```

The canonical step order is **forward → loss → backward → step → zero_grad**.
Gradients are never cleared automatically; skipping `zero_grad()` accumulates
them.

### Serialization

```python
from core import io

io.save_model(net, "model.mpk")
restored = io.load_model("model.mpk")   # returns a rebuilt Sequential
```

The format is a tagged text file (`minipytorch-v2` header) covering `Linear`,
`Conv2d`, and all stateless layers.

### Current limitations

- **No batching**: `Linear` takes 1-D inputs, `Conv2d`/pooling take a single
  unbatched `(C, H, W)` sample. Train one sample at a time.
- **No autograd**: `backward` is explicit and per-module; there is no
  computation graph.
- `Conv2d`/pooling accept **square stride/padding only** (native kernel
  limitation).
- `Linear`'s default activation is **sigmoid**, not identity, pass an
  explicit `activation` argument to change it.
- The extension targets **x86-64 Linux** (the assembly kernels use GNU
  assembler syntax and the System V AMD64 ABI).

## Assembly core

Numerical kernels target x86-64 Linux using GNU assembler syntax and the
System V AMD64 ABI. C++ owns vectors, object lifetime, validation, exceptions,
serialization, and high-level orchestration; assembly receives only raw
buffers, scalar values, and sizes. Layer forward passes and parameter updates
use the assembly kernels throughout.

Assembly sigmoid and tanh use bounded scalar approximations. Their documented
maximum absolute error is `1e-6` on the normal input ranges; tests compare
kernel outputs against the C++ reference implementations. Keep assembly
changes narrowly scoped: add or alter one kernel, its C++ bridge, direct
regression tests, and benchmark evidence in a single commit.

## Requirements

- `uv` (creates the venv and performs editable installs)
- `g++` or any C++17-capable compiler, plus `make` (C++ tests/experiments)
- Python >= 3.9 with `numpy` (see `requirements.txt`)
- `cmake` >= 3.15 and `pybind11` >= 2.13 for building the extension by hand
  (`./run.sh setup` fetches these automatically in its isolated build env)
- x86-64 Linux for the assembly kernels
