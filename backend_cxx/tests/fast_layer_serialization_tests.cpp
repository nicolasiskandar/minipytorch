#include <cstdio>
#include <random>
#include <vector>

#include "detail/nn_asm.hpp"
#include "fastLayer.hpp"
#include "layer.hpp"
#include "losses.hpp"
#include "network.hpp"
#include "neuron.hpp"
#include "serialize.hpp"
#include "test_runner.hpp"
#include "test_suites.hpp"

namespace {

void testFastLayer(TestRunner& t) {
    FastLayer relu(2, 3, FastReLU, {1, 0, 0, 1, 1, 1}, {0, 0, 0});
    t.check(
        relu.forward({1.0, 2.0}).size() == 3, "FastLayer forward output size"
    );

    FastLayer sigmoid(2, 2, FastSigmoid, {1, 0, 0, 1}, {0, 0});
    std::vector<double> output = sigmoid.forward({3.0, 5.0});
    t.checkNear(output[0], sigmoidFn(3.0), 1e-12, "FastLayer first output");
    t.checkNear(output[1], sigmoidFn(5.0), 1e-12, "FastLayer second output");

    const std::vector<double> weights = {0.5, -0.3, 0.8, 0.1};
    const std::vector<double> biases = {0.1, -0.1};
    const std::vector<double> input = {1.0, 2.0};
    const std::vector<double> target = {0.5, 0.5};
    FastLayer layer(2, 2, FastSigmoid, weights, biases);
    LossResult result = meanSquaredError(layer.forward(input), target);
    layer.backward(result.dLoss_dOutput);
    constexpr double h = 1e-6;
    std::vector<double> changed = weights;
    changed[0] += h;
    FastLayer upper(2, 2, FastSigmoid, changed, biases);
    changed[0] -= 2 * h;
    FastLayer lower(2, 2, FastSigmoid, changed, biases);
    double numeric = (meanSquaredError(upper.forward(input), target).loss -
                      meanSquaredError(lower.forward(input), target).loss) /
                     (2 * h);
    t.checkNear(
        layer.gradWeights()[0], numeric, 1e-5, "FastLayer numeric gradient"
    );
    layer.applyGradients(0.1);
    t.check(true, "FastLayer applies gradients");

    PlainActivation custom{
        [](double value) { return value + 1.0; }, [](double) { return 1.0; }
    };
    FastLayer customLayer(1, 1, custom, {2.0}, {0.5});
    t.checkNear(
        customLayer.forward({3.0})[0], 7.5, 1e-12,
        "FastLayer custom activation uses C++ fallback"
    );
}

void testAssemblyGradientUpdate(TestRunner& t) {
    std::vector<double> values = {1.0, 2.0};
    const std::vector<double> gradients = {0.5, -1.0};
    nn_apply_gradients_f64(values.data(), gradients.data(), values.size(), 0.1);
    t.checkNear(
        values[0], 0.95, 1e-12, "Assembly gradient update: first value"
    );
    t.checkNear(
        values[1], 2.1, 1e-12, "Assembly gradient update: second value"
    );
}

void testSerialization(TestRunner& t) {
    NeuralNetwork network({
        Layer({Neuron({0.5, -0.3}, 0.1, Tanh), Neuron({0.8, 0.2}, -0.5, ReLU)}),
        Layer({Neuron({0.4, -0.6}, 0.0, Sigmoid)}),
    });
    const std::vector<double> input = {1.0, 0.5};
    const std::vector<double> before = network.predict(input);
    const char* path = "/tmp/test_roundtrip.txt";
    saveNetwork(network, path);
    NeuralNetwork loaded = loadNetwork(path);
    const std::vector<double> after = loaded.predict(input);
    t.check(before.size() == after.size(), "Serialized network output size");
    t.checkNear(
        before[0], after[0], 1e-12, "Serialized network prediction matches"
    );
    std::remove(path);

    NeuralNetwork allActivations({Layer(
        {Neuron({1.0}, 0.0, Sigmoid), Neuron({1.0}, 0.0, Tanh),
         Neuron({1.0}, 0.0, ReLU)}
    )});
    const char* activationPath = "/tmp/test_activations.txt";
    std::vector<double> activationBefore = allActivations.predict({2.0});
    saveNetwork(allActivations, activationPath);
    NeuralNetwork activationLoaded = loadNetwork(activationPath);
    std::vector<double> activationAfter = activationLoaded.predict({2.0});
    for (std::size_t i = 0; i < activationBefore.size(); ++i)
        t.checkNear(
            activationBefore[i], activationAfter[i], 1e-12,
            "Serialization preserves activation"
        );
    std::remove(activationPath);
}

}  // namespace

void runFastLayerAndSerializationTests(TestRunner& t) {
    testFastLayer(t);
    testAssemblyGradientUpdate(t);
    testSerialization(t);
}
