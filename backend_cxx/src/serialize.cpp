#include "serialize.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::string activationName(const Activation& activation) {
    if (auto target = activation.forward.target<double (*)(double)>()) {
        if (*target == sigmoidFn) return "sigmoid";
        if (*target == tanhFn) return "tanh";
        if (*target == reluFn) return "relu";
    }
    throw std::runtime_error(
        "activationName: unrecognized activation function"
    );
}

Activation activationFromName(const std::string& name) {
    if (name == "sigmoid") return Sigmoid;
    if (name == "tanh") return Tanh;
    if (name == "relu") return ReLU;
    throw std::runtime_error(
        "activationFromName: unknown activation '" + name + "'"
    );
}

void saveNetwork(const NeuralNetwork& net, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("saveNetwork: cannot open '" + filename + "'");

    const auto& layers = net.layers();
    out << layers.size() << "\n";

    for (size_t l = 0; l < layers.size(); ++l) {
        const Layer& layer = layers[l];
        out << layer.size() << "\n";

        for (size_t n = 0; n < layer.size(); ++n) {
            const Neuron& neuron = layer.neuron(n);
            out << activationName(neuron.activation()) << " ";
            out << std::setprecision(17) << neuron.bias();

            const auto& w = neuron.weights();
            for (size_t i = 0; i < w.size(); ++i)
                out << " " << std::setprecision(17) << w[i];
            out << "\n";
        }
    }
}

static std::vector<double> parseWeights(std::istringstream& iss) {
    std::vector<double> weights;
    std::string token;
    while (std::getline(iss, token, ' ')) {
        if (token.empty()) continue;
        weights.push_back(std::stod(token));
    }
    return weights;
}

NeuralNetwork loadNetwork(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("loadNetwork: cannot open '" + filename + "'");

    std::string line;

    std::getline(in, line);
    size_t numLayers = std::stoul(line);

    std::vector<Layer> layers;
    layers.reserve(numLayers);

    for (size_t l = 0; l < numLayers; ++l) {
        std::getline(in, line);
        size_t numNeurons = std::stoul(line);

        std::vector<Neuron> neurons;
        neurons.reserve(numNeurons);

        for (size_t n = 0; n < numNeurons; ++n) {
            std::getline(in, line);
            std::istringstream iss(line);

            std::string actName;
            std::getline(iss, actName, ' ');

            std::string biasStr;
            std::getline(iss, biasStr, ' ');
            double bias = std::stod(biasStr);

            std::vector<double> weights = parseWeights(iss);

            neurons.emplace_back(
                std::move(weights), bias, activationFromName(actName)
            );
        }

        layers.emplace_back(std::move(neurons));
    }

    return NeuralNetwork(std::move(layers));
}
