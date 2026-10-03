#pragma once
#include <vector>

#include "activations.hpp"

class Neuron {
   public:
    Neuron(std::vector<double> weights, double bias, Activation activation)
        : weights_(std::move(weights)),
          bias_(bias),
          activation_(std::move(activation)) {}

    double forward(const std::vector<double>& input);
    std::vector<double> backward(double dLoss_dOutput);
    void applyGradients(double learningRate);

    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }
    const Activation& activation() const { return activation_; }
    const std::vector<double>& gradWeights() const { return gradWeights_; }

   private:
    std::vector<double> weights_;
    double bias_;
    Activation activation_;

    std::vector<double> lastInput_;
    double lastOutput_ = 0.0;
    std::vector<double> gradWeights_;
    double gradBias_ = 0.0;
};
