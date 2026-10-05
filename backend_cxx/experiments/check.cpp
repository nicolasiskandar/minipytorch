#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include "fastLayer.hpp"

namespace {

constexpr int kIterations = 2000;

// Opaque so FastLayer takes the C++ fallback instead of the builtin kind.
const PlainActivation kOpaqueTanh{tanhFn, tanhDerivFromOutput};

double milliseconds(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double maxAbsDifference(const std::vector<double>& left,
                        const std::vector<double>& right) {
    double worst = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i)
        worst = std::max(worst, std::abs(left[i] - right[i]));
    return worst;
}

}  // namespace

int main() {
    const std::size_t numInputs = 128, numOutputs = 128;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> weights(numOutputs * numInputs);
    std::vector<double> biases(numOutputs);
    std::vector<double> input(numInputs);
    for (double& weight : weights) weight = dist(rng);
    for (double& bias : biases) bias = dist(rng);
    for (double& value : input) value = dist(rng);

    FastLayer assemblyLayer(numInputs, numOutputs, FastTanh, weights, biases);
    FastLayer fallbackLayer(numInputs, numOutputs, kOpaqueTanh, weights, biases);

    const std::vector<double> assemblyOutput = assemblyLayer.forward(input);
    const std::vector<double> fallbackOutput = fallbackLayer.forward(input);
    std::cout << "max forward output difference: "
              << maxAbsDifference(assemblyOutput, fallbackOutput) << "\n";

    const std::vector<double> dLoss_dOutput(numOutputs, 1.0);
    const std::vector<double> assemblyDInput =
        assemblyLayer.backward(dLoss_dOutput);
    const std::vector<double> fallbackDInput =
        fallbackLayer.backward(dLoss_dOutput);
    std::cout << "max backward dLoss_dInput difference: "
              << maxAbsDifference(assemblyDInput, fallbackDInput) << "\n"
              << "max gradWeights difference: "
              << maxAbsDifference(
                     assemblyLayer.gradWeights(), fallbackLayer.gradWeights()
                 )
              << "\n";

    auto start = std::chrono::steady_clock::now();
    for (int it = 0; it < kIterations; ++it) assemblyLayer.forward(input);
    auto mid = std::chrono::steady_clock::now();
    for (int it = 0; it < kIterations; ++it) fallbackLayer.forward(input);
    auto end = std::chrono::steady_clock::now();

    std::cout << "forward x" << kIterations << " (" << numInputs << "x"
              << numOutputs << ")\n"
              << "  fused asm kernel : " << milliseconds(start, mid) << " ms\n"
              << "  per-unit fallback: " << milliseconds(mid, end) << " ms\n";

    start = std::chrono::steady_clock::now();
    for (int it = 0; it < kIterations; ++it) assemblyLayer.backward(dLoss_dOutput);
    mid = std::chrono::steady_clock::now();
    for (int it = 0; it < kIterations; ++it) fallbackLayer.backward(dLoss_dOutput);
    end = std::chrono::steady_clock::now();

    std::cout << "backward x" << kIterations << "\n"
              << "  fused asm kernel : " << milliseconds(start, mid) << " ms\n"
              << "  per-unit fallback: " << milliseconds(mid, end) << " ms\n";

    return 0;
}
