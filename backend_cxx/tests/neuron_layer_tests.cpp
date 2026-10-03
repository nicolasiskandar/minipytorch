#include <stdexcept>
#include <vector>

#include "detail/nn_asm.hpp"
#include "layer.hpp"
#include "losses.hpp"
#include "neuron.hpp"
#include "test_runner.hpp"
#include "test_suites.hpp"

namespace {

void testNeuronForward(TestRunner& t) {
    Neuron neuron({0.5, -0.5}, 0.0, ReLU);
    t.checkNear(
        neuron.forward({2.0, 1.0}), 0.5, 1e-9,
        "Neuron forward includes weights and activation"
    );

    Neuron biased({1.0, 1.0}, 0.5, Sigmoid);
    t.checkNear(
        biased.forward({1.0, 1.0}), sigmoidFn(2.5), 1e-12,
        "Neuron forward includes bias"
    );

    bool threw = false;
    try {
        neuron.forward({1.0, 2.0, 3.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    t.check(threw, "Neuron forward rejects mismatched input size");
    Neuron negative({1.0, 1.0}, 0.0, ReLU);
    t.checkNear(
        negative.forward({-1.0, -1.0}), 0.0, 1e-12,
        "Neuron forward clamps negative ReLU output"
    );
}

void testAssemblyNeuronForward(TestRunner& t) {
    const std::vector<double> input = {2.0, -1.0};
    const std::vector<double> weights = {0.5, 1.0};
    t.checkNear(
        nn_neuron_forward_f64(
            input.data(), weights.data(), weights.size(), 0.25,
            NN_ACTIVATION_SIGMOID
        ),
        nn_sigmoid_f64(0.25), 1e-12, "Assembly neuron forward"
    );
}

void testNeuronBackwardAndUpdate(TestRunner& t) {
    Neuron neuron({0.5, -0.5}, 0.0, Sigmoid);
    double output = neuron.forward({2.0, 1.0});
    neuron.backward(meanSquaredError({output}, {1.0}).dLoss_dOutput[0]);
    t.checkNear(
        neuron.gradWeights()[0], -0.177447, 1e-5, "Neuron backward weight 0"
    );
    t.checkNear(
        neuron.gradWeights()[1], -0.088723, 1e-5, "Neuron backward weight 1"
    );
    t.check(
        neuron.backward(1.0).size() == 2,
        "Neuron backward returns input-sized gradient"
    );

    double oldWeight = neuron.weights()[0];
    neuron.applyGradients(0.5);
    t.checkNear(
        neuron.weights()[0], oldWeight - 0.5 * neuron.gradWeights()[0], 1e-12,
        "Neuron applies gradient"
    );
}

void testNeuronGradientCheck(TestRunner& t) {
    constexpr double h = 1e-6;
    auto lossForWeight = [&](double weight) {
        Neuron probe({weight, -0.5}, 0.0, Sigmoid);
        return meanSquaredError({probe.forward({2.0, 1.0})}, {1.0}).loss;
    };
    Neuron neuron({0.5, -0.5}, 0.0, Sigmoid);
    double output = neuron.forward({2.0, 1.0});
    neuron.backward(meanSquaredError({output}, {1.0}).dLoss_dOutput[0]);
    double numeric =
        (lossForWeight(0.5 + h) - lossForWeight(0.5 - h)) / (2 * h);
    t.checkNear(
        neuron.gradWeights()[0], numeric, 1e-4,
        "Neuron analytic gradient matches numeric gradient"
    );
}

void testLayer(TestRunner& t) {
    Layer layer(
        {Neuron({1.0, 0.0}, 0.0, ReLU), Neuron({0.0, 1.0}, 0.0, ReLU),
         Neuron({1.0, 1.0}, 0.0, ReLU)}
    );
    std::vector<double> output = layer.forward({1.0, 2.0});
    t.check(output.size() == 3, "Layer forward output size");
    t.checkNear(output[2], 3.0, 1e-12, "Layer forward output value");
    std::vector<double> inputGradient = layer.backward({1.0, 1.0, 1.0});
    t.check(inputGradient.size() == 2, "Layer backward input size");
    layer.applyGradients(0.1);
    t.check(true, "Layer applies gradients");
}

void testAssemblyAccumulation(TestRunner& t) {
    std::vector<double> destination = {1.0, -2.0, 0.5};
    const std::vector<double> source = {2.0, 3.0, -0.5};
    nn_accumulate_f64(destination.data(), source.data(), destination.size());
    t.checkNear(
        destination[0], 3.0, 1e-12, "Assembly accumulation: first element"
    );
    t.checkNear(
        destination[1], 1.0, 1e-12, "Assembly accumulation: second element"
    );
    t.checkNear(
        destination[2], 0.0, 1e-12, "Assembly accumulation: third element"
    );
}

}  // namespace

void runNeuronAndLayerTests(TestRunner& t) {
    testNeuronForward(t);
    testAssemblyNeuronForward(t);
    testNeuronBackwardAndUpdate(t);
    testNeuronGradientCheck(t);
    testLayer(t);
    testAssemblyAccumulation(t);
}
