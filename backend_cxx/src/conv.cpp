#include "conv.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "detail/nn_asm.hpp"

namespace nn {

Conv2DResult conv2d_forward(
    const std::vector<double>& input,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    int inC, int inH, int inW,
    int outC, int kH, int kW,
    int stride,
    int padding)
{
    Conv2DResult res;
    res.stride = stride;
    res.padding = padding;
    res.inputShape = {inC, inH, inW};
    res.weightShape = {outC, inC, kH, kW};

    int outH = (inH + 2 * padding - kH) / stride + 1;
    int outW = (inW + 2 * padding - kW) / stride + 1;
    if (outH < 1) outH = 1;
    if (outW < 1) outW = 1;
    res.outputShape = {outC, outH, outW};

    // im2col: for each output position, collect inC*kH*kW values
    // build col buffer of size (inC*kH*kW) * (outH*outW)
    std::size_t colRows = static_cast<std::size_t>(inC) * static_cast<std::size_t>(kH) * static_cast<std::size_t>(kW);
    std::size_t colCols = static_cast<std::size_t>(outH) * static_cast<std::size_t>(outW);
    res.colBuffer.assign(colRows * colCols, 0.0);
    double* col = res.colBuffer.data();

    // fill col
    std::size_t colIdx = 0;
    for (int oh = 0; oh < outH; ++oh) {
        for (int ow = 0; ow < outW; ++ow) {
            int hStart = oh * stride - padding;
            int wStart = ow * stride - padding;
            // fill one column: all kernel positions across all channels
            for (int ic = 0; ic < inC; ++ic) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int ih = hStart + kh;
                        int iw = wStart + kw;
                        double val = 0.0;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            std::size_t inIdx = static_cast<std::size_t>(ic) * inH * inW + static_cast<std::size_t>(ih) * inW + static_cast<std::size_t>(iw);
                            val = input[inIdx];
                        } else {
                            val = 0.0;
                        }
                        col[colIdx++] = val;
                    }
                }
            }
        }
    }

    // now for each output channel, compute outC_{c} = weight_c (flattened, length colRows) dot col
    res.output.assign(static_cast<std::size_t>(outC) * colCols, 0.0);
    double* outPtr = res.output.data();
    for (int oc = 0; oc < outC; ++oc) {
        const double* wrow = weight.data() + static_cast<std::size_t>(oc) * colRows;
        // dot wrow with each column of col: result goes to outPtr[oc * colCols + j]
        for (std::size_t j = 0; j < colCols; ++j) {
            const double* ccol = col + j * colRows; // column j
            double s = nn_dot_product_f64(wrow, ccol, colRows);
            outPtr[static_cast<std::size_t>(oc) * colCols + j] = s + (bias.empty() ? 0.0 : bias[oc]);
        }
    }
    return res;
}

std::vector<double> conv2d_backward_input(
    const std::vector<double>& gradOutput,
    const std::vector<double>& weight,
    int inC, int inH, int inW,
    int outC, int kH, int kW,
    int stride,
    int padding)
{
    int outH = (inH + 2 * padding - kH) / stride + 1;
    int outW = (inW + 2 * padding - kW) / stride + 1;
    if (outH < 1) outH = 1;
    if (outW < 1) outW = 1;
    std::size_t colRows = static_cast<std::size_t>(inC) * kH * kW;
    std::size_t colCols = static_cast<std::size_t>(outH) * outW;
    // compute gradCol: weight^T @ gradOutput (each col of gradCol is sum_c gradOutput_c * w_c)
    std::vector<double> gradCol(colRows * colCols, 0.0);
    double* gCol = gradCol.data();
    for (std::size_t j = 0; j < colCols; ++j) {
        // initialize column j to zero
        std::fill(gCol + j * colRows, gCol + (j + 1) * colRows, 0.0);
        for (int oc = 0; oc < outC; ++oc) {
            double go_ij = gradOutput[static_cast<std::size_t>(oc) * colCols + j];
            if (go_ij == 0.0) continue;
            const double* wrow = weight.data() + static_cast<std::size_t>(oc) * colRows;
            // accumulate go_ij * wrow into gCol column j
            for (std::size_t k = 0; k < colRows; ++k) {
                gCol[j * colRows + k] += go_ij * wrow[k];
            }
        }
    }
    // col2im
    std::vector<double> gradInput(static_cast<std::size_t>(inC) * inH * inW, 0.0);
    std::size_t colIdx = 0;
    // need to iterate positions in same order as forward - iterate oh,ow then fill
    for (int oh = 0; oh < outH; ++oh) {
        for (int ow = 0; ow < outW; ++ow) {
            int hStart = oh * stride - padding;
            int wStart = ow * stride - padding;
            for (int ic = 0; ic < inC; ++ic) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int ih = hStart + kh;
                        int iw = wStart + kw;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            gradInput[static_cast<std::size_t>(ic) * inH * inW + static_cast<std::size_t>(ih) * inW + static_cast<std::size_t>(iw)] += gradCol[colIdx];
                        }
                        ++colIdx;
                    }
                }
            }
        }
    }
    return gradInput;
}

void conv2d_backward_weight(
    const std::vector<double>& gradOutput,
    const std::vector<double>& input,
    std::vector<double>& gradWeight,
    std::vector<double>& gradBias,
    int inC, int inH, int inW,
    int outC, int kH, int kW,
    int stride,
    int padding)
{
    int outH = (inH + 2 * padding - kH) / stride + 1;
    int outW = (inW + 2 * padding - kW) / stride + 1;
    if (outH < 1) outH = 1;
    if (outW < 1) outW = 1;
    std::size_t colRows = static_cast<std::size_t>(inC) * kH * kW;
    std::size_t colCols = static_cast<std::size_t>(outH) * outW;
    // build col
    std::vector<double> col(colRows * colCols, 0.0);
    double* cptr = col.data();
    std::size_t colIdx = 0;
    for (int oh = 0; oh < outH; ++oh) {
        for (int ow = 0; ow < outW; ++ow) {
            int hStart = oh * stride - padding;
            int wStart = ow * stride - padding;
            for (int ic = 0; ic < inC; ++ic) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int ih = hStart + kh;
                        int iw = wStart + kw;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            cptr[colIdx++] = input[static_cast<std::size_t>(ic) * inH * inW + static_cast<std::size_t>(ih) * inW + static_cast<std::size_t>(iw)];
                        } else {
                            cptr[colIdx++] = 0.0;
                        }
                    }
                }
            }
        }
    }
    // gradWeight[oc, :] += sum_j gradOutput[oc,j] * col[:,j]^T? 
    // gradWeight is (outC * colRows): dW_oc,k = sum_{oh,ow} gradOutput[oc,oh,ow] * X_col_k,oh,ow
    gradWeight.assign(static_cast<std::size_t>(outC) * colRows, 0.0);
    double* gwPtr = gradWeight.data();
    for (int oc = 0; oc < outC; ++oc) {
        for (std::size_t j = 0; j < colCols; ++j) {
            double go = gradOutput[static_cast<std::size_t>(oc) * colCols + j];
            if (go == 0.0) continue;
            const double* ccol = cptr + j * colRows;
            for (std::size_t k = 0; k < colRows; ++k) {
                gwPtr[static_cast<std::size_t>(oc) * colRows + k] += go * ccol[k];
            }
        }
    }
    // gradBias
    gradBias.assign(outC, 0.0);
    for (int oc = 0; oc < outC; ++oc) {
        double s = 0.0;
        for (std::size_t j = 0; j < colCols; ++j) {
            s += gradOutput[static_cast<std::size_t>(oc) * colCols + j];
        }
        gradBias[oc] = s;
    }
}

} // namespace nn
