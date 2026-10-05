#pragma once

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "fastLayer.hpp"
#include "losses.hpp"

using LossFn = LossResult (*)(
    const std::vector<double>& predicted,
    const std::vector<double>& target
);

class ExperimentMlp {
   public:
    ExperimentMlp(
        std::size_t numInputs,
        std::size_t hidden,
        std::size_t numOutputs,
        std::vector<double> weights1,
        std::vector<double> biases1,
        std::vector<double> weights2,
        std::vector<double> biases2
    )
        : layer1_(numInputs, hidden, FastTanh, std::move(weights1), std::move(biases1)),
          layer2_(hidden, numOutputs, FastSigmoid, std::move(weights2), std::move(biases2)) {}

    std::vector<double> predict(const std::vector<double>& input) {
        return layer2_.forward(layer1_.forward(input));
    }

    double trainStep(
        const std::vector<double>& input,
        const std::vector<double>& target,
        double learningRate,
        LossFn loss
    ) {
        std::vector<double> hidden = layer1_.forward(input);
        std::vector<double> output = layer2_.forward(hidden);
        LossResult result = loss(output, target);
        std::vector<double> dHidden = layer2_.backward(result.dLoss_dOutput);
        layer1_.backward(dHidden);
        layer2_.applyGradients(learningRate);
        layer1_.applyGradients(learningRate);
        return result.loss;
    }

   private:
    FastLayer layer1_;
    FastLayer layer2_;
};

// Uniform(-1, 1) weights for a (inputs x outputs) layer, C-order per output unit.
inline std::vector<double> randomLayerWeights(
    std::mt19937& rng,
    std::uniform_real_distribution<double>& dist,
    std::size_t inputs,
    std::size_t outputs
) {
    std::vector<double> weights(inputs * outputs);
    for (double& weight : weights) weight = dist(rng);
    return weights;
}
