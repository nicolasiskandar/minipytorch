#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "activations.hpp"
#include "detail/nn_asm.hpp"
#include "test_runner.hpp"
#include "test_suites.hpp"

namespace {

void testCppActivations(TestRunner& t) {
    t.checkNear(sigmoidFn(0.0), 0.5, 1e-12, "Sigmoid(0) = 0.5");
    t.checkNear(sigmoidFn(1.0), 0.731058, 1e-5, "Sigmoid(1) ~ 0.73106");
    t.checkNear(sigmoidFn(-1.0), 0.268941, 1e-5, "Sigmoid(-1) ~ 0.26894");
    t.checkNear(sigmoidFn(10.0), 0.999954, 1e-5, "Sigmoid(10) ~ 0.99995");
    t.checkNear(sigmoidFn(-10.0), 0.00004539, 1e-7, "Sigmoid(-10) ~ 0");
    t.checkNear(
        sigmoidDerivFromOutput(sigmoidFn(1.0)),
        sigmoidFn(1.0) * (1.0 - sigmoidFn(1.0)), 1e-12, "Sigmoid derivative"
    );
    t.checkNear(tanhFn(-1.0), -0.761594, 1e-5, "Tanh(-1) ~ -0.76159");
    t.checkNear(tanhFn(0.0), 0.0, 1e-12, "Tanh(0) = 0");
    t.checkNear(tanhFn(10.0), 0.999999, 1e-5, "Tanh(10) ~ 1");
    t.checkNear(
        tanhDerivFromOutput(tanhFn(1.0)), 1.0 - tanhFn(1.0) * tanhFn(1.0),
        1e-12, "Tanh derivative"
    );
    t.checkNear(reluFn(-3.0), 0.0, 1e-12, "ReLU clamps negative input");
    t.checkNear(reluFn(5.0), 5.0, 1e-12, "ReLU preserves positive input");
    t.checkNear(reluDerivFromOutput(5.0), 1.0, 1e-12, "ReLU derivative");
    t.checkNear(
        reluDerivFromOutput(0.0), 0.0, 1e-12, "ReLU derivative at zero"
    );
}

void testAssemblyDotProduct(TestRunner& t) {
    const std::vector<double> left = {2.0, -1.0, 0.5};
    const std::vector<double> right = {3.0, 4.0, 2.0};
    t.checkNear(
        nn_dot_product_f64(left.data(), right.data(), left.size()), 3.0, 1e-12,
        "Assembly dot product: mixed-sign values"
    );
    t.checkNear(
        nn_dot_product_f64(nullptr, nullptr, 0), 0.0, 1e-12,
        "Assembly dot product: empty input"
    );
}

void testAssemblyActivations(TestRunner& t) {
    t.checkNear(nn_relu_f64(-3.0), 0.0, 1e-12, "Assembly ReLU: negative input");
    t.checkNear(nn_relu_f64(2.5), 2.5, 1e-12, "Assembly ReLU: positive input");
    t.checkNear(
        nn_relu_derivative_from_output_f64(2.5), 1.0, 1e-12,
        "Assembly ReLU derivative"
    );
    for (double input : {-16.0, -8.0, -1.0, 0.0, 1.0, 8.0, 16.0}) {
        double expected = std::exp(input);
        t.checkNear(
            nn_exp_approx_f64(input), expected,
            std::max(1e-12, expected * 1e-9), "Assembly exp approximation"
        );
    }
    for (double input : {-14.0, -8.0, -1.0, 0.0, 1.0, 8.0, 14.0})
        t.checkNear(
            nn_sigmoid_f64(input), sigmoidFn(input), 1e-6,
            "Assembly sigmoid approximation"
        );
    for (double input : {-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0})
        t.checkNear(
            nn_tanh_f64(input), tanhFn(input), 1e-6,
            "Assembly tanh approximation"
        );
    t.check(
        std::isnan(nn_sigmoid_f64(std::numeric_limits<double>::quiet_NaN())),
        "Assembly sigmoid preserves NaN"
    );
}

void testAssemblyFastLayerForward(TestRunner& t) {
    const std::vector<double> input = {2.0, -1.0};
    const std::vector<double> weights = {0.5, 1.0, -1.0, 0.25};
    const std::vector<double> biases = {0.25, -0.5};
    std::vector<double> output(2);
    nn_fast_layer_forward_f64(
        input.data(), weights.data(), biases.data(), output.data(), 2, 2,
        NN_ACTIVATION_TANH
    );
    t.checkNear(
        output[0], nn_tanh_f64(0.25), 1e-12,
        "Assembly FastLayer forward: output 0"
    );
    t.checkNear(
        output[1], nn_tanh_f64(-2.75), 1e-12,
        "Assembly FastLayer forward: output 1"
    );
}

}  // namespace

void runActivationAndKernelTests(TestRunner& t) {
    testCppActivations(t);
    testAssemblyDotProduct(t);
    testAssemblyActivations(t);
    testAssemblyFastLayerForward(t);
}
