#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include "fastLayer.hpp"
#include "layer.hpp"
#include "neuron.hpp"

int main() {
    const std::size_t numInputs = 5, numOutputs = 4;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> flatWeights(numOutputs * numInputs);
    std::vector<double> biases(numOutputs);
    std::vector<Neuron> neurons;

    for (std::size_t o = 0; o < numOutputs; ++o) {
        std::vector<double> w(numInputs);
        for (std::size_t i = 0; i < numInputs; ++i) {
            w[i] = dist(rng);
            flatWeights[o * numInputs + i] = w[i];
        }
        double b = dist(rng);
        biases[o] = b;
        neurons.push_back(Neuron(w, b, Tanh));
    }

    Layer layer(neurons);
    FastLayer fastLayer(numInputs, numOutputs, FastTanh, flatWeights, biases);

    std::vector<double> input(numInputs);
    for (auto& v : input) v = dist(rng);

    // auto out1 = layer.forward(input);
    // auto out2 = fastLayer.forward(input);
    // double maxDiff = 0.0;
    // for (std::size_t i = 0; i < out1.size(); ++i)
    //     maxDiff = std::max(maxDiff, std::abs(out1[i] - out2[i]));
    // std::cout << "max forward output difference: " << maxDiff << "\n";

    // std::vector<double> dOut(numOutputs, 1.0);
    // auto dIn1 = layer.backward(dOut);
    // auto dIn2 = fastLayer.backward(dOut);
    // double maxDiffB = 0.0;
    // for (std::size_t i = 0; i < dIn1.size(); ++i)
    //     maxDiffB = std::max(maxDiffB, std::abs(dIn1[i] - dIn2[i]));
    // std::cout << "max backward dLoss_dInput difference: " << maxDiffB <<
    // "\n";

    layer.forward(input);
    fastLayer.forward(input);

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < 2000; ++it) layer.forward(input);
    auto t1 = std::chrono::steady_clock::now();
    for (int it = 0; it < 2000; ++it) fastLayer.forward(input);
    auto t2 = std::chrono::steady_clock::now();

    double msLayer = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double msFast = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "Layer: " << msLayer << std::endl;
    std::cout << "Fast Layer: " << msFast << std::endl;

    return 0;
}