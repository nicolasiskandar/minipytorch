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
