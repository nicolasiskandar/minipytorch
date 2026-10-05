#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "detail/nn_asm.hpp"
#include "fastLayer.hpp"
#include "losses.hpp"
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

std::vector<double> randomValues(std::mt19937& rng, std::size_t count) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> values(count);
    for (double& value : values) value = dist(rng);
    return values;
}

// Two FastLayers wired by hand, replacing NeuralNetwork::trainStep. Dimensions
// are explicit, never derived from a buffer moved in the same call: that reads
// size 0 and builds a layer with numOutputs == 0.
class MiniNet {
   public:
    MiniNet(
        std::size_t numInputs,
        std::size_t hidden,
        std::size_t numOutputs,
        PlainActivation hiddenActivation,
        std::vector<double> w1,
        std::vector<double> b1,
        std::vector<double> w2,
        std::vector<double> b2
    )
        : l1_(numInputs, hidden, hiddenActivation, std::move(w1), std::move(b1)),
          l2_(hidden, numOutputs, FastSigmoid, std::move(w2), std::move(b2)) {}

    static MiniNet random(
        std::size_t numInputs,
        std::size_t hidden,
        std::size_t numOutputs,
        PlainActivation hiddenActivation,
        std::mt19937& rng
    ) {
        return MiniNet(
            numInputs, hidden, numOutputs, hiddenActivation,
            randomValues(rng, numInputs * hidden), randomValues(rng, hidden),
            randomValues(rng, hidden * numOutputs),
            randomValues(rng, numOutputs)
        );
    }

    std::vector<double> predict(const std::vector<double>& input) {
        return l2_.forward(l1_.forward(input));
    }

    double
    trainStep(const std::vector<double>& input, const std::vector<double>& target, double lr) {
        std::vector<double> hidden = l1_.forward(input);
        std::vector<double> output = l2_.forward(hidden);
        LossResult result = meanSquaredError(output, target);
        std::vector<double> dHidden = l2_.backward(result.dLoss_dOutput);
        l1_.backward(dHidden);
        l2_.applyGradients(lr);
        l1_.applyGradients(lr);
        return result.loss;
    }

   private:
    FastLayer l1_;
    FastLayer l2_;
};

void testFastLayerTraining(TestRunner& t) {
    std::mt19937 rng(2);
    MiniNet xorNet = MiniNet::random(2, 2, 1, FastTanh, rng);
    const std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    const std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
    for (int epoch = 0; epoch < 8000; ++epoch)
        for (std::size_t i = 0; i < inputs.size(); ++i)
            xorNet.trainStep(inputs[i], targets[i], 0.8);

    t.check(xorNet.predict({0, 0})[0] < 0.1, "XOR predicts zero for (0,0)");
    t.check(xorNet.predict({0, 1})[0] > 0.9, "XOR predicts one for (0,1)");
    t.check(xorNet.predict({1, 0})[0] > 0.9, "XOR predicts one for (1,0)");
    t.check(xorNet.predict({1, 1})[0] < 0.1, "XOR predicts zero for (1,1)");

    MiniNet single = MiniNet::random(2, 1, 1, FastTanh, rng);
    double initial = single.trainStep({1.0, 1.0}, {1.0}, 0.5);
    for (int i = 0; i < 2000; ++i) single.trainStep({1.0, 1.0}, {1.0}, 0.5);
    double final = single.trainStep({1.0, 1.0}, {1.0}, 0.5);
    t.check(final < initial, "Network loss decreases during training");
    t.check(single.predict({1.0, 1.0})[0] > 0.4, "Single hidden unit learns target");
}

void testFastLayerMultiOutputTraining(TestRunner& t) {
    // 2 hidden units x 2 inputs, then 2 output units x 2 hidden.
    MiniNet identity(2, 2, 2, FastReLU, {0.5, -0.2, -0.1, 0.5}, {0.0, 0.0},
                     {0.4, -0.3, -0.2, 0.4}, {0.0, 0.0});
    for (int epoch = 0; epoch < 1000; ++epoch) {
        identity.trainStep({1.0, 0.0}, {1.0, 0.0}, 0.5);
        identity.trainStep({0.0, 1.0}, {0.0, 1.0}, 0.5);
    }
    std::vector<double> output = identity.predict({1.0, 0.0});
    t.check(output[0] > 0.6, "Identity network retains first signal");
    t.check(
        output[1] < 0.4, "Identity network suppresses second signal"
    );
}

}  // namespace

void runLossTests(TestRunner& t) {
    testLosses(t);
    testAssemblyMse(t);
    testFastLayerTraining(t);
    testFastLayerMultiOutputTraining(t);
}
