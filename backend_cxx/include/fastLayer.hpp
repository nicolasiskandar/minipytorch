#pragma once
#include <vector>

#include "activations.hpp"

using PlainActivationFn = double (*)(double);

struct PlainActivation {
    PlainActivationFn forward;
    PlainActivationFn derivativeFromOutput;
};

extern const PlainActivation FastSigmoid;
extern const PlainActivation FastTanh;
extern const PlainActivation FastReLU;

class FastLayer {
   public:
    FastLayer(
        std::size_t numInputs,
        std::size_t numOutputs,
        PlainActivation activation,
        std::vector<double> weights,
        std::vector<double> biases
    )
        : numInputs_(numInputs),
          numOutputs_(numOutputs),
          activation_(activation),
          weights_(std::move(weights)),
          biases_(std::move(biases)) {}

    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& dLoss_dOutput);
    void applyGradients(double learningRate);

    const std::vector<double>& gradWeights() const { return gradWeights_; }
    const std::vector<double>& gradBiases() const { return gradBiases_; }
    const std::vector<double>& weights() const { return weights_; }
    const std::vector<double>& biases() const { return biases_; }
    void setWeights(const std::vector<double>& w) { weights_ = w; }
    void setBiases(const std::vector<double>& b) { biases_ = b; }

   private:
    std::size_t numInputs_;
    std::size_t numOutputs_;
    PlainActivation activation_;
    std::vector<double> weights_;
    std::vector<double> biases_;
    std::vector<double> lastInput_;
    std::vector<double> lastOutput_;
    std::vector<double> gradWeights_;
    std::vector<double> gradBiases_;
};

extern const PlainActivation FastReLU6;

extern const PlainActivation FastLeakyReLU;
