#include <cmath>
#include <vector>

#include "conv.hpp"
#include "test_runner.hpp"
#include "test_suites.hpp"

namespace {

void testConv2dForward(TestRunner& t) {
    // 1x1x3x3 input, one 2x2 kernel, stride 1, no padding -> 2x2 output
    const std::vector<double> input = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const std::vector<double> weight = {1.0, 1.0, 1.0, 1.0};
    const std::vector<double> bias = {0.0};
    auto res = nn::conv2d_forward(input, weight, bias, 1, 3, 3, 1, 2, 2, 1, 0);
    t.check(
        res.outputShape.size() == 3 && res.outputShape[0] == 1 &&
            res.outputShape[1] == 2 && res.outputShape[2] == 2,
        "Conv2d forward output shape"
    );
    t.checkNear(res.output[0], 8.0, 1e-12, "Conv2d forward window 0");
    t.checkNear(res.output[1], 12.0, 1e-12, "Conv2d forward window 1");
    t.checkNear(res.output[2], 20.0, 1e-12, "Conv2d forward window 2");
    t.checkNear(res.output[3], 24.0, 1e-12, "Conv2d forward window 3");
}

void testConv2dBiasAndStride(TestRunner& t) {
    const std::vector<double> input(25, 1.0);  // 1x5x5
    const std::vector<double> weight(9, 0.0);
    const std::vector<double> bias = {2.5};
    auto res = nn::conv2d_forward(input, weight, bias, 1, 5, 5, 1, 3, 3, 1, 0);
    t.check(
        res.outputShape[1] == 3 && res.outputShape[2] == 3,
        "Conv2d 5x5 with 3x3 kernel gives 3x3"
    );
    t.checkNear(res.output[0], 2.5, 1e-12, "Conv2d bias is added per output");

    auto strided =
        nn::conv2d_forward(input, weight, bias, 1, 5, 5, 1, 3, 3, 2, 0);
    t.check(
        strided.outputShape[1] == 2 && strided.outputShape[2] == 2,
        "Conv2d stride 2 halves the output"
    );
}

void testConv2dPaddingZeroesBorder(TestRunner& t) {
    // A single 1x1 kernel with padding 1 must reproduce the input exactly:
    // the padded border reads as zero, so it only proves the shape is right.
    const std::vector<double> input = {1.0, 2.0, 3.0};
    const std::vector<double> weight = {2.0};
    const std::vector<double> bias = {0.0};
    auto res = nn::conv2d_forward(input, weight, bias, 1, 1, 3, 1, 1, 1, 1, 1);
    t.check(
        res.outputShape[1] == 3 && res.outputShape[2] == 5,
        "Conv2d padded 1x1 kernel pads both spatial dims"
    );
    // The real input row survives at output row 1, columns 1..3.
    t.checkNear(
        res.output[1 * 5 + 1], 2.0, 1e-12, "Conv2d padded row keeps real values"
    );
    t.checkNear(
        res.output[1 * 5 + 2], 4.0, 1e-12, "Conv2d padded row scales values"
    );
    t.checkNear(
        res.output[0], 0.0, 1e-12, "Conv2d fully padded row is zero"
    );
    t.checkNear(
        res.output[2 * 5 + 0], 0.0, 1e-12, "Conv2d padded column is zero"
    );

    // Now a kernel that reaches into the zero padding. A 1x3 kernel padded by
    // 1 turns a single row into three output rows, all three of which share
    // the same input row at different horizontal offsets.
    const std::vector<double> input2 = {1.0, 1.0, 1.0};
    const std::vector<double> weight2 = {1.0, 1.0, 1.0};
    auto res2 = nn::conv2d_forward(input2, weight2, bias, 1, 1, 3, 1, 1, 3, 1, 1);
    t.check(
        res2.outputShape[1] == 3 && res2.outputShape[2] == 3,
        "Conv2d padded 1x3 kernel expands the row count"
    );
    t.checkNear(
        res2.output[3], 2.0, 1e-12, "Conv2d left padding contributes zero"
    );
    t.checkNear(
        res2.output[4], 3.0, 1e-12, "Conv2d unpadded window sums all three"
    );
    t.checkNear(
        res2.output[5], 2.0, 1e-12, "Conv2d right padding contributes zero"
    );
    t.checkNear(res2.output[1], 0.0, 1e-12, "Conv2d fully padded window is zero");
}

void testConv2dBackwardMatchesNumeric(TestRunner& t) {
    // 1x2x3 input with a 2x2 kernel, stride 1, no padding -> 1x2 output
    const std::vector<double> input = {0.5, -1.0, 2.0, 0.25, 1.5, -0.5};
    const std::vector<double> weight = {0.3, -0.2, 0.1, 0.4};
    const std::vector<double> bias = {0.05};
    const std::vector<double> gradOutput = {1.0, -2.0};

    auto res = nn::conv2d_forward(input, weight, bias, 1, 2, 3, 1, 2, 2, 1, 0);

    std::vector<double> gradWeight;
    std::vector<double> gradBias;
    nn::conv2d_backward_weight(
        gradOutput, input, gradWeight, gradBias, 1, 2, 3, 1, 2, 2, 1, 0
    );

    // Numeric weight gradient: perturb each weight and re-measure the output.
    constexpr double h = 1e-6;
    for (std::size_t i = 0; i < weight.size(); ++i) {
        std::vector<double> plus = weight;
        std::vector<double> minus = weight;
        plus[i] += h;
        minus[i] -= h;
        auto up = nn::conv2d_forward(input, plus, bias, 1, 2, 3, 1, 2, 2, 1, 0);
        auto down = nn::conv2d_forward(input, minus, bias, 1, 2, 3, 1, 2, 2, 1, 0);
        double numeric = 0.0;
        for (std::size_t j = 0; j < gradOutput.size(); ++j)
            numeric += gradOutput[j] * (up.output[j] - down.output[j]) / (2 * h);
        t.checkNear(
            gradWeight[i], numeric, 1e-5, "Conv2d numeric weight gradient"
        );
    }

    double numericBias = 0.0;
    {
        std::vector<double> plus = bias;
        std::vector<double> minus = bias;
        plus[0] += h;
        minus[0] -= h;
        auto up = nn::conv2d_forward(input, weight, plus, 1, 2, 3, 1, 2, 2, 1, 0);
        auto down = nn::conv2d_forward(input, weight, minus, 1, 2, 3, 1, 2, 2, 1, 0);
        for (std::size_t j = 0; j < gradOutput.size(); ++j)
            numericBias += gradOutput[j] * (up.output[j] - down.output[j]) / (2 * h);
    }
    t.checkNear(gradBias[0], numericBias, 1e-5, "Conv2d numeric bias gradient");

    auto gradInput = nn::conv2d_backward_input(
        gradOutput, weight, 1, 2, 3, 1, 2, 2, 1, 0
    );
    for (std::size_t i = 0; i < input.size(); ++i) {
        std::vector<double> plus = input;
        std::vector<double> minus = input;
        plus[i] += h;
        minus[i] -= h;
        auto up = nn::conv2d_forward(plus, weight, bias, 1, 2, 3, 1, 2, 2, 1, 0);
        auto down =
            nn::conv2d_forward(minus, weight, bias, 1, 2, 3, 1, 2, 2, 1, 0);
        double numeric = 0.0;
        for (std::size_t j = 0; j < gradOutput.size(); ++j)
            numeric += gradOutput[j] * (up.output[j] - down.output[j]) / (2 * h);
        t.checkNear(
            gradInput[i], numeric, 1e-5, "Conv2d numeric input gradient"
        );
    }
}

}  // namespace

void runConvTests(TestRunner& t) {
    testConv2dForward(t);
    testConv2dBiasAndStride(t);
    testConv2dPaddingZeroesBorder(t);
    testConv2dBackwardMatchesNumeric(t);
}