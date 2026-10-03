#pragma once
#include <vector>

#include "neuron.hpp"

class Layer {
   public:
    explicit Layer(std::vector<Neuron> neurons)
        : neurons_(std::move(neurons)) {}

    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& dLoss_dOutput);
    void applyGradients(double learningRate);

    std::size_t size() const { return neurons_.size(); }
    const Neuron& neuron(std::size_t i) const { return neurons_[i]; }

   private:
    std::vector<Neuron> neurons_;
    std::size_t inputSize_ = 0;
};