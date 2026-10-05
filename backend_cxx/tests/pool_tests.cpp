#include <cmath>
#include <vector>

#include "pool.hpp"
#include "test_runner.hpp"
#include "test_suites.hpp"

namespace {

void testMaxPool2d(TestRunner& t) {
    const std::vector<double> input = {
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0,
        12.0, 13.0, 14.0, 15.0,
    };
    auto res = nn::maxpool2d_forward(input, 1, 4, 4, 2, -1, 0);
    t.check(
        res.outputShape[1] == 2 && res.outputShape[2] == 2,
        "MaxPool2d halves the spatial size"
    );
    t.checkNear(res.output[0], 5.0, 1e-12, "MaxPool2d output 0");
    t.checkNear(res.output[1], 7.0, 1e-12, "MaxPool2d output 1");
    t.checkNear(res.output[2], 13.0, 1e-12, "MaxPool2d output 2");
    t.checkNear(res.output[3], 15.0, 1e-12, "MaxPool2d output 3");
    t.check(
        res.indices[0] == 5 && res.indices[1] == 7 && res.indices[2] == 13 &&
            res.indices[3] == 15,
        "MaxPool2d records the argmax index"
    );

    const std::vector<double> gradOutput = {1.0, 1.0, 1.0, 1.0};
    auto gradInput = nn::maxpool2d_backward(gradOutput, res.indices, 1, 4, 4, 1, 2, 2);
    for (std::size_t i = 0; i < gradInput.size(); ++i) {
        bool isArgmax = (i == 5 || i == 7 || i == 13 || i == 15);
        t.checkNear(
            gradInput[i], isArgmax ? 1.0 : 0.0, 1e-12,
            "MaxPool2d routes gradient only to the argmax"
        );
    }
}

void testAvgPool2d(TestRunner& t) {
    const std::vector<double> input = {
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0,
        12.0, 13.0, 14.0, 15.0,
    };
    auto res = nn::avgpool2d_forward(input, 1, 4, 4, 2, -1, 0);
    t.checkNear(res.output[0], 2.5, 1e-12, "AvgPool2d averages the window");
    t.checkNear(res.output[1], 4.5, 1e-12, "AvgPool2d output 1");
    t.checkNear(res.output[2], 10.5, 1e-12, "AvgPool2d output 2");
    t.checkNear(res.output[3], 12.5, 1e-12, "AvgPool2d output 3");

    const std::vector<double> gradOutput = {1.0, 0.0, 0.0, 0.0};
    auto gradInput = nn::avgpool2d_backward(gradOutput, 1, 4, 4, 2, 2, 0, 1, 2, 2);
    // The first window covers input indices 0, 1, 4, 5 in (C, H, W) order.
    t.checkNear(gradInput[0], 0.25, 1e-12, "AvgPool2d spreads gradient to (0,0)");
    t.checkNear(gradInput[1], 0.25, 1e-12, "AvgPool2d spreads gradient to (0,1)");
    t.checkNear(gradInput[4], 0.25, 1e-12, "AvgPool2d spreads gradient to (1,0)");
    t.checkNear(gradInput[5], 0.25, 1e-12, "AvgPool2d spreads gradient to (1,1)");
    t.checkNear(gradInput[2], 0.0, 1e-12, "AvgPool2d leaves other windows alone");
    t.checkNear(gradInput[6], 0.0, 1e-12, "AvgPool2d leaves later windows alone");
}

void testPoolingStridesAndPadding(TestRunner& t) {
    const std::vector<double> input(25, 1.0);  // 1x5x5
    auto overlapping = nn::maxpool2d_forward(input, 1, 5, 5, 2, 1, 0);
    t.check(
        overlapping.outputShape[1] == 4 && overlapping.outputShape[2] == 4,
        "MaxPool2d stride 1 overlaps windows"
    );
    auto padded = nn::maxpool2d_forward(input, 1, 5, 5, 3, 1, 1);
    t.check(
        padded.outputShape[1] == 5 && padded.outputShape[2] == 5,
        "MaxPool2d padding keeps the input size"
    );
    auto avg = nn::avgpool2d_forward(input, 1, 5, 5, 3, 1, 1);
    t.check(
        avg.outputShape[1] == 5 && avg.outputShape[2] == 5,
        "AvgPool2d padding keeps the input size"
    );
}

}  // namespace

void runPoolTests(TestRunner& t) {
    testMaxPool2d(t);
    testAvgPool2d(t);
    testPoolingStridesAndPadding(t);
}