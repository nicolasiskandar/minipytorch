#include "layer.hpp"

#include "detail/nn_asm.hpp"

std::vector<double> Layer::forward(const std::vector<double>& input) {
    inputSize_ = input.size();
    std::vector<double> output;
    output.reserve(neurons_.size());
    for (Neuron& neuron : neurons_) output.push_back(neuron.forward(input));
    return output;
}

std::vector<double> Layer::backward(const std::vector<double>& dLoss_dOutput) {
    std::vector<double> dLoss_dInput(inputSize_, 0.0);
    for (std::size_t i = 0; i < neurons_.size(); ++i) {
        std::vector<double> dIn = neurons_[i].backward(dLoss_dOutput[i]);
        nn_accumulate_f64(dLoss_dInput.data(), dIn.data(), dIn.size());
    }
    return dLoss_dInput;
}

void Layer::applyGradients(double learningRate) {
    for (Neuron& neuron : neurons_) neuron.applyGradients(learningRate);
}
