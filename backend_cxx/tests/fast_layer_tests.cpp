#include <cstddef>
#include <vector>

#include "detail/nn_asm.hpp"
#include "fastLayer.hpp"
#include "losses.hpp"
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

void testFastLayerZeroGradients(TestRunner& t) {
    const std::vector<double> weights = {0.5, -0.3, 0.8, 0.2};
    const std::vector<double> biases = {0.1, -0.5};
    FastLayer layer(2, 2, FastSigmoid, weights, biases);
    LossResult result = meanSquaredError(layer.forward({1.0, 2.0}), {0.5, 0.5});
    layer.backward(result.dLoss_dOutput);

    bool anyNonZero = false;
    for (double g : layer.gradWeights()) anyNonZero = anyNonZero || g != 0.0;
    for (double g : layer.gradBiases()) anyNonZero = anyNonZero || g != 0.0;
    t.check(anyNonZero, "FastLayer gradients are populated before zeroing");

    layer.zeroGradients();
    t.checkNear(
        layer.gradWeights()[0], 0.0, 1e-12, "FastLayer zeroGradients: weight 0"
    );
    t.checkNear(
        layer.gradWeights()[3], 0.0, 1e-12, "FastLayer zeroGradients: weight 3"
    );
    t.checkNear(
        layer.gradBiases()[0], 0.0, 1e-12, "FastLayer zeroGradients: bias 0"
    );
    t.checkNear(
        layer.gradBiases()[1], 0.0, 1e-12, "FastLayer zeroGradients: bias 1"
    );

    layer.applyGradients(0.5);
    t.checkNear(
        layer.weights()[0], weights[0], 1e-12,
        "FastLayer zeroGradients leaves weights untouched"
    );
}

void testFastLayerApplyGradientsBeforeBackward(TestRunner& t) {
    const std::vector<double> weights = {0.5, -0.3, 0.8, 0.2};
    const std::vector<double> biases = {0.1, -0.5};
    FastLayer layer(2, 2, FastSigmoid, weights, biases);
    t.checkNear(
        layer.gradWeights()[0], 0.0, 1e-12,
        "FastLayer gradients start zeroed before any backward"
    );
    layer.applyGradients(0.5);
    t.checkNear(
        layer.weights()[0], weights[0], 1e-12,
        "FastLayer applyGradients before backward leaves weights untouched"
    );
    t.checkNear(
        layer.biases()[1], biases[1], 1e-12,
        "FastLayer applyGradients before backward leaves biases untouched"
    );
}

}  // namespace

void runFastLayerTests(TestRunner& t) {
    testFastLayer(t);
    testAssemblyGradientUpdate(t);
    testFastLayerZeroGradients(t);
    testFastLayerApplyGradientsBeforeBackward(t);
}
