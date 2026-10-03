#include <iostream>
#include <random>

#include "layer.hpp"
#include "losses.hpp"
#include "network.hpp"
#include "neuron.hpp"

int main() {
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    NeuralNetwork net({
      Layer({
          Neuron({dist(rng), dist(rng)}, dist(rng), Tanh),
          Neuron({dist(rng), dist(rng)}, dist(rng), Tanh)
      }),
      Layer({Neuron({dist(rng), dist(rng)}, dist(rng), Sigmoid)})
    });

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    for (int epoch = 0; epoch < 8000; ++epoch)
        for (std::size_t i = 0; i < inputs.size(); ++i)
            net.trainStep(inputs[i], targets[i], 0.8, meanSquaredError);

    for (std::size_t i = 0; i < inputs.size(); ++i)
        std::cout << "XOR(" << inputs[i][0] << "," << inputs[i][1] << ") -> "
                  << net.predict(inputs[i])[0] << "\n";

    return 0;
}