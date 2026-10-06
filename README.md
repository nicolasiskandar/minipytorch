# Neural Network

A neural network from scratch with no external dependencies in C++ and Assembly

## Building

Requires `g++` (or any C++17-capable compiler).

```sh
make          # build everything (tests, experiments, bench)
make test     # run the test suite
make run      # XOR experiment
make circle   # circle classification experiment
make bench    # FastLayer vs Layer benchmark
make clean    # remove binaries
```

## Python API

The Python bindings (`core/`) wrap the C++ backend. The compiled extension and
the prebuilt C++ binaries are part of a build tree, so a plain environment
(no `pip install`) is enough to run the Python tests:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

./run.sh test-py    # Python test suite
./run.sh test-all   # C++ and Python tests
```

`core/` imports the extension `_minipytorch`, which must exist under
`.venv/lib/python*/site-packages/core/` or `PYTHONPATH` must include the build
tree.

## Project Structure

```
include/           Public headers
  neuron.hpp       Neuron (weights, bias, activation)
  layer.hpp        Layer (vector of neurons)
  fastLayer.hpp    FastLayer (flat-buffer layer, function-pointer activations)
  network.hpp      NeuralNetwork (chains layers, forward/backward/train)
  activations.hpp  Sigmoid, Tanh, ReLU (std::function-based)
  losses.hpp        Mean Squared Error, Binary Cross Entropy
  serialize.hpp    Save/load network weights to text files
  detail/           Internal assembly-kernel ABI declarations
src/               C++ implementations (one module per subsystem)
asm/kernels/       x86-64 scalar assembly numerical kernels
tests/             Test runner and subsystem-specific test modules
experiments/       XOR, circle classification, benchmark
```

## Assembly Core

Numerical kernels target x86-64 Linux using GNU assembler syntax and the
System V AMD64 ABI. C++ owns vectors, object lifetime, validation, exceptions,
serialization, and high-level network orchestration; assembly receives only
raw buffers, scalar values, and sizes. Built-in `FastLayer` and `Neuron`
activations use the assembly kernels, while user-supplied activation callbacks
use the C++ fallback.

Assembly sigmoid and tanh use bounded scalar approximations. Their documented
maximum absolute error is `1e-6` on the normal input ranges; tests compare
kernel outputs against the C++ reference implementations. Keep assembly
changes narrowly scoped: add or alter one kernel, its C++ bridge, direct
regression tests, and benchmark evidence in a single commit.

## Usage

```cpp
#include "network.hpp"
#include "losses.hpp"

NeuralNetwork net({
    Layer({
        Neuron({0.5, -0.3}, 0.1, Tanh),
        Neuron({0.8, 0.2}, -0.5, Tanh)
    }),
    Layer({Neuron({0.4, -0.6}, 0.0, Sigmoid)})
});

// Train
net.trainStep({0.0, 1.0}, {1.0}, 0.8, meanSquaredError);

// Predict
std::vector<double> out = net.predict({0.0, 1.0});

// Save / load
saveNetwork(net, "model.txt");
NeuralNetwork loaded = loadNetwork("model.txt");
```

## Activation Functions

| Name    | Forward       | Derivative (from output) |
|---------|---------------|--------------------------|
| Sigmoid | 1 / (1 + e^-z) | y * (1 - y)            |
| Tanh    | tanh(z)       | 1 - y^2                  |
| ReLU    | max(0, z)     | y > 0 ? 1 : 0           |

## Loss Functions

- **Mean Squared Error** (`meanSquaredError`): general purpose regression
- **Binary Cross Entropy** (`binaryCrossEntropy`): binary classification

## Serialization

Networks can be saved and loaded from plain text files:

```cpp
saveNetwork(net, "model.txt");
NeuralNetwork loaded = loadNetwork("model.txt");
```

The format stores layer count, neuron count per layer, and for each neuron the activation name, bias, and all weights.
