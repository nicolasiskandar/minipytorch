#include "pool.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace nn {

MaxPool2DResult maxpool2d_forward(
    const std::vector<double>& input,
    int inC, int inH, int inW,
    int kernelSize,
    int stride,
    int padding)
{
    if (stride < 0) stride = kernelSize;
    MaxPool2DResult res;
    res.kernelSize = kernelSize;
    res.stride = stride;
    res.padding = padding;
    res.inputShape = {inC, inH, inW};

    int outH = (inH + 2 * padding - kernelSize) / stride + 1;
    int outW = (inW + 2 * padding - kernelSize) / stride + 1;
    if (outH < 1) outH = 1;
    if (outW < 1) outW = 1;
    res.outputShape = {inC, outH, outW};

    std::size_t outCount = static_cast<std::size_t>(inC) * outH * outW;
    res.output.assign(outCount, -std::numeric_limits<double>::infinity());
    res.indices.assign(outCount, -1);

    for (int c = 0; c < inC; ++c) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                int hStart = oh * stride - padding;
                int wStart = ow * stride - padding;
                double best = -std::numeric_limits<double>::infinity();
                int bestIdx = -1;
                for (int kh = 0; kh < kernelSize; ++kh) {
                    for (int kw = 0; kw < kernelSize; ++kw) {
                        int ih = hStart + kh;
                        int iw = wStart + kw;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            std::size_t inIdx = static_cast<std::size_t>(c) * inH * inW + static_cast<std::size_t>(ih) * inW + static_cast<std::size_t>(iw);
                            if (bestIdx == -1 || input[inIdx] > best) {
                                best = input[inIdx];
                                bestIdx = static_cast<int>(inIdx);
                            }
                        } else {
                            if (bestIdx == -1 || -std::numeric_limits<double>::infinity() > best) {
                                best = -std::numeric_limits<double>::infinity();
                                bestIdx = -1;
                            }
                        }
                    }
                }
                std::size_t outIdx = static_cast<std::size_t>(c) * outH * outW + static_cast<std::size_t>(oh) * outW + static_cast<std::size_t>(ow);
                res.output[outIdx] = best;
                res.indices[outIdx] = bestIdx;
            }
        }
    }
    return res;
}

std::vector<double> maxpool2d_backward(
    const std::vector<double>& gradOutput,
    const std::vector<int>& indices,
    int inC, int inH, int inW,
    int outC, int outH, int outW)
{
    std::vector<double> gradInput(static_cast<std::size_t>(inC) * inH * inW, 0.0);
    std::size_t outCount = static_cast<std::size_t>(outC) * outH * outW;
    for (std::size_t i = 0; i < outCount; ++i) {
        int inIdx = indices[i];
        if (inIdx >= 0) {
            gradInput[static_cast<std::size_t>(inIdx)] += gradOutput[i];
        }
    }
    return gradInput;
}

AvgPool2DResult avgpool2d_forward(
    const std::vector<double>& input,
    int inC, int inH, int inW,
    int kernelSize,
    int stride,
    int padding)
{
    if (stride < 0) stride = kernelSize;
    AvgPool2DResult res;
    res.kernelSize = kernelSize;
    res.stride = stride;
    res.padding = padding;
    res.inputShape = {inC, inH, inW};

    int outH = (inH + 2 * padding - kernelSize) / stride + 1;
    int outW = (inW + 2 * padding - kernelSize) / stride + 1;
    if (outH < 1) outH = 1;
    if (outW < 1) outW = 1;
    res.outputShape = {inC, outH, outW};

    std::size_t outCount = static_cast<std::size_t>(inC) * outH * outW;
    res.output.assign(outCount, 0.0);
    double area = static_cast<double>(kernelSize * kernelSize);

    for (int c = 0; c < inC; ++c) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                int hStart = oh * stride - padding;
                int wStart = ow * stride - padding;
                double sum = 0.0;
                int count = 0;
                for (int kh = 0; kh < kernelSize; ++kh) {
                    for (int kw = 0; kw < kernelSize; ++kw) {
                        int ih = hStart + kh;
                        int iw = wStart + kw;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            std::size_t inIdx = static_cast<std::size_t>(c) * inH * inW + static_cast<std::size_t>(ih) * inW + static_cast<std::size_t>(iw);
                            sum += input[inIdx];
                            ++count;
                        }
                    }
                }
                std::size_t outIdx = static_cast<std::size_t>(c) * outH * outW + static_cast<std::size_t>(oh) * outW + static_cast<std::size_t>(ow);
                res.output[outIdx] = sum / (count > 0 ? static_cast<double>(count) : area);
            }
        }
    }
    return res;
}

std::vector<double> avgpool2d_backward(
    const std::vector<double>& gradOutput,
    int inC,
    int inH,
    int inW,
    int kernelSize,
    int stride,
    int padding,
    int /*outC*/,
    int outH,
    int outW)
{
    std::vector<double> gradInput(static_cast<std::size_t>(inC) * inH * inW, 0.0);
    for (int c = 0; c < inC; ++c) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                int hStart = oh * stride - padding;
                int wStart = ow * stride - padding;
                int count = 0;
                for (int kh = 0; kh < kernelSize; ++kh) {
                    for (int kw = 0; kw < kernelSize; ++kw) {
                        int ih = hStart + kh;
                        int iw = wStart + kw;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            ++count;
                        }
                    }
                }
                if (count == 0) continue;
                std::size_t outIdx = static_cast<std::size_t>(c) * outH * outW + static_cast<std::size_t>(oh) * outW + static_cast<std::size_t>(ow);
                double grad = gradOutput[outIdx] / static_cast<double>(count);
                for (int kh = 0; kh < kernelSize; ++kh) {
                    for (int kw = 0; kw < kernelSize; ++kw) {
                        int ih = hStart + kh;
                        int iw = wStart + kw;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            gradInput[static_cast<std::size_t>(c) * inH * inW + static_cast<std::size_t>(ih) * inW + static_cast<std::size_t>(iw)] += grad;
                        }
                    }
                }
            }
        }
    }
    return gradInput;
}

} // namespace nn
