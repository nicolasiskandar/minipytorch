#include "activations.hpp"

#include <algorithm>
#include <cmath>

double sigmoidFn(double z) { return 1.0 / (1.0 + std::exp(-z)); }

double sigmoidDerivFromOutput(double y) { return y * (1.0 - y); }

double tanhFn(double z) { return std::tanh(z); }

double tanhDerivFromOutput(double y) { return 1.0 - y * y; }

double reluFn(double z) { return std::max(0.0, z); }

double reluDerivFromOutput(double y) { return y > 0.0 ? 1.0 : 0.0; }

const Activation Sigmoid{sigmoidFn, sigmoidDerivFromOutput};
const Activation Tanh{tanhFn, tanhDerivFromOutput};
const Activation ReLU{reluFn, reluDerivFromOutput};

double relu6Fn(double z) {
    if (z < 0.0) return 0.0;
    if (z > 6.0) return 6.0;
    return z;
}

double relu6DerivFromOutput(double y) {
    if (y <= 0.0 || y >= 6.0) return 0.0;
    return 1.0;
}

const Activation ReLU6{relu6Fn, relu6DerivFromOutput};

double leakyReluFn(double z) { return z > 0.0 ? z : 0.01 * z; }

double leakyReluDerivFromOutput(double y) { return y > 0.0 ? 1.0 : 0.01; }

const Activation LeakyReLU{leakyReluFn, leakyReluDerivFromOutput};
