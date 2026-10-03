#include <cmath>
#include <random>
#include <vector>

#include "detail/nn_asm.hpp"
#include "layer.hpp"
#include "losses.hpp"
#include "network.hpp"
#include "neuron.hpp"
#include "test_runner.hpp"
#include "test_suites.hpp"

namespace {

void testLosses(TestRunner& t) {
    LossResult mse = meanSquaredError({1.0, 3.0}, {2.0, 5.0});
    t.checkNear(mse.loss, 2.5, 1e-12, "MSE multi-output loss");
    t.checkNear(mse.dLoss_dOutput[0], -1.0, 1e-12, "MSE first gradient");
    t.checkNear(mse.dLoss_dOutput[1], -2.0, 1e-12, "MSE second gradient");

    LossResult bce = binaryCrossEntropy({0.9}, {1.0});
    t.checkNear(bce.loss, 0.10536, 1e-4, "BCE loss");
    t.checkNear(bce.dLoss_dOutput[0], -1.11111, 1e-4, "BCE gradient");
    t.checkNear(
        binaryCrossEntropy({0.9}, {0.0}).loss, -std::log(1.0 - 0.9), 1e-5,
        "BCE target-zero loss"
    );
    constexpr double bceH = 1e-7;
    double bceNumeric = (binaryCrossEntropy({0.8 + bceH}, {1.0}).loss -
                         binaryCrossEntropy({0.8 - bceH}, {1.0}).loss) /
                        (2 * bceH);
    t.checkNear(
        binaryCrossEntropy({0.8}, {1.0}).dLoss_dOutput[0], bceNumeric, 1e-4,
        "BCE numeric gradient"
    );

    constexpr double h = 1e-7;
    double numeric = (meanSquaredError({2.0 + h}, {3.0}).loss -
                      meanSquaredError({2.0 - h}, {3.0}).loss) /
                     (2 * h);
    t.checkNear(
        meanSquaredError({2.0}, {3.0}).dLoss_dOutput[0], numeric, 1e-5,
        "MSE numeric gradient"
    );
}

void testAssemblyMse(TestRunner& t) {
    const std::vector<double> predicted = {1.0, 3.0};
    const std::vector<double> target = {2.0, 5.0};
    std::vector<double> gradient(2);
    double loss = 0.0;
    nn_mean_squared_error_f64(
        predicted.data(), target.data(), gradient.data(), predicted.size(),
        &loss
    );
    t.checkNear(loss, 2.5, 1e-12, "Assembly MSE loss");
    t.checkNear(gradient[0], -1.0, 1e-12, "Assembly MSE first gradient");
    t.checkNear(gradient[1], -2.0, 1e-12, "Assembly MSE second gradient");
}

NeuralNetwork makeXorNetwork(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return NeuralNetwork({
        Layer(
            {Neuron({dist(rng), dist(rng)}, dist(rng), Tanh),
             Neuron({dist(rng), dist(rng)}, dist(rng), Tanh)}
        ),
        Layer({Neuron({dist(rng), dist(rng)}, dist(rng), Sigmoid)}),
    });
}

void testNetworkTraining(TestRunner& t) {
    NeuralNetwork network = makeXorNetwork(2);
    const std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    const std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    for (int epoch = 0; epoch < 8000; ++epoch)
        for (std::size_t i = 0; i < inputs.size(); ++i)
            network.trainStep(inputs[i], targets[i], 0.8, meanSquaredError);

    t.check(network.predict({0, 0})[0] < 0.1, "XOR predicts zero for (0,0)");
    t.check(network.predict({0, 1})[0] > 0.9, "XOR predicts one for (0,1)");
    t.check(network.predict({1, 0})[0] > 0.9, "XOR predicts one for (1,0)");
    t.check(network.predict({1, 1})[0] < 0.1, "XOR predicts zero for (1,1)");

    NeuralNetwork single({Layer({Neuron({0.1, 0.1}, 0.0, Sigmoid)})});
    double initial = single.trainStep({1.0, 1.0}, {1.0}, 0.5, meanSquaredError);
    for (int i = 0; i < 2000; ++i)
        single.trainStep({1.0, 1.0}, {1.0}, 0.5, meanSquaredError);
    double final = single.trainStep({1.0, 1.0}, {1.0}, 0.5, meanSquaredError);
    t.check(final < initial, "Network loss decreases during training");
    t.check(single.predict({1.0, 1.0})[0] > 0.9, "Single neuron learns target");

    NeuralNetwork identity({
        Layer({Neuron({0.5, -0.2}, 0.0, ReLU), Neuron({-0.1, 0.5}, 0.0, ReLU)}),
        Layer(
            {Neuron({0.4, -0.3}, 0.0, Sigmoid),
             Neuron({-0.2, 0.4}, 0.0, Sigmoid)}
        ),
    });
    for (int epoch = 0; epoch < 1000; ++epoch) {
        identity.trainStep({1.0, 0.0}, {1.0, 0.0}, 0.5, meanSquaredError);
        identity.trainStep({0.0, 1.0}, {0.0, 1.0}, 0.5, meanSquaredError);
    }
    std::vector<double> identityOutput = identity.predict({1.0, 0.0});
    t.check(identityOutput[0] > 0.6, "Identity network retains first signal");
    t.check(
        identityOutput[1] < 0.4, "Identity network suppresses second signal"
    );
}

}  // namespace

void runLossAndNetworkTests(TestRunner& t) {
    testLosses(t);
    testAssemblyMse(t);
    testNetworkTraining(t);
}
