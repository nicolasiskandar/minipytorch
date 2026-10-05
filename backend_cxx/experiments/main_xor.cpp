#include <iostream>
#include <random>

#include "losses.hpp"
#include "mlp.hpp"

int main() {
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    ExperimentMlp net(
        2, 2, 1,
        randomLayerWeights(rng, dist, 2, 2), randomLayerWeights(rng, dist, 1, 2),
        randomLayerWeights(rng, dist, 2, 1), randomLayerWeights(rng, dist, 1, 1)
    );

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
