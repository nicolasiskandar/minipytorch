#include "network.hpp"

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> current = input;
    for (Layer& layer : layers_) current = layer.forward(current);
    return current;
}

double NeuralNetwork::trainStep(
    const std::vector<double>& input,
    const std::vector<double>& target,
    double learningRate,
    const LossFn& lossFn
) {
    std::vector<double> output = predict(input);
    LossResult result = lossFn(output, target);

    std::vector<double> grad = result.dLoss_dOutput;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        grad = it->backward(grad);

    for (Layer& layer : layers_) layer.applyGradients(learningRate);

    return result.loss;
}
