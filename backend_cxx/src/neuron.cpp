#include "neuron.hpp"

#include <stdexcept>

#include "detail/nn_asm.hpp"

namespace {

int assemblyActivationKind(const Activation& activation) {
    auto target = activation.forward.target<double (*)(double)>();
    if (target == nullptr) return -1;
    if (*target == sigmoidFn) return NN_ACTIVATION_SIGMOID;
    if (*target == tanhFn) return NN_ACTIVATION_TANH;
    if (*target == reluFn) return NN_ACTIVATION_RELU;
    return -1;
}

}  // namespace

double Neuron::forward(const std::vector<double>& input) {
    if (input.size() != weights_.size())
        throw std::invalid_argument(
            "Neuron::forward: input size does not match weight count"
        );

    lastInput_ = input;

    int activationKind = assemblyActivationKind(activation_);
    if (activationKind >= 0) {
        lastOutput_ = nn_neuron_forward_f64(
            input.data(), weights_.data(), weights_.size(), bias_,
            activationKind
        );
        return lastOutput_;
    }

    double z = bias_;
    for (std::size_t i = 0; i < weights_.size(); ++i)
        z += weights_[i] * input[i];

    lastOutput_ = activation_.forward(z);

    return lastOutput_;
}

std::vector<double> Neuron::backward(double dLoss_dOutput) {
    gradWeights_.assign(weights_.size(), 0.0);
    std::vector<double> dLoss_dInput(weights_.size());
    int activationKind = assemblyActivationKind(activation_);
    if (activationKind >= 0) {
        nn_neuron_backward_f64(
            weights_.data(), lastInput_.data(), weights_.size(), lastOutput_,
            dLoss_dOutput, activationKind, gradWeights_.data(), &gradBias_,
            dLoss_dInput.data()
        );
        return dLoss_dInput;
    }

    double dOutput_dZ = activation_.derivativeFromOutput(lastOutput_);
    double delta = dLoss_dOutput * dOutput_dZ;

    for (std::size_t i = 0; i < weights_.size(); ++i)
        gradWeights_[i] = delta * lastInput_[i];

    gradBias_ = delta;

    for (std::size_t i = 0; i < weights_.size(); ++i)
        dLoss_dInput[i] = delta * weights_[i];

    return dLoss_dInput;
}

void Neuron::applyGradients(double learningRate) {
    nn_apply_gradients_f64(
        weights_.data(), gradWeights_.data(), weights_.size(), learningRate
    );
    nn_apply_gradients_f64(&bias_, &gradBias_, 1, learningRate);
}
