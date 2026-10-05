#pragma once
#include <vector>

namespace nn {

struct MaxPool2DResult {
    std::vector<double> output;
    std::vector<int> indices; // argmax indices in input (flattened) for each output element
    std::vector<int> inputShape;
    std::vector<int> outputShape;
    int kernelSize;
    int stride;
    int padding;
};

MaxPool2DResult maxpool2d_forward(
    const std::vector<double>& input,
    int inC, int inH, int inW,
    int kernelSize,
    int stride = -1,
    int padding = 0
);

std::vector<double> maxpool2d_backward(
    const std::vector<double>& gradOutput,
    const std::vector<int>& indices,
    int inC, int inH, int inW,
    int outC, int outH, int outW
);

struct AvgPool2DResult {
    std::vector<double> output;
    std::vector<int> inputShape;
    std::vector<int> outputShape;
    int kernelSize;
    int stride;
    int padding;
};

AvgPool2DResult avgpool2d_forward(
    const std::vector<double>& input,
    int inC, int inH, int inW,
    int kernelSize,
    int stride = -1,
    int padding = 0
);

std::vector<double> avgpool2d_backward(
    const std::vector<double>& gradOutput,
    int inC, int inH, int inW,
    int kernelSize,
    int stride,
    int padding,
    int outC, int outH, int outW
);

} // namespace nn
