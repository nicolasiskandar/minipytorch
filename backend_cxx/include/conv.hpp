#pragma once
#include <vector>

namespace nn {

struct Conv2DResult {
    std::vector<double> output;
    std::vector<double> colBuffer;
    std::vector<int> inputShape; // C,H,W
    std::vector<int> weightShape; // OC,IC,KH,KW
    std::vector<int> outputShape; // OC,OH,OW
    int stride;
    int padding;
};

Conv2DResult conv2d_forward(
    const std::vector<double>& input,      // C_in*H_in*W_in
    const std::vector<double>& weight,     // outC*inC*kH*kW
    const std::vector<double>& bias,       // outC (optional, size 0 if none)
    int inC, int inH, int inW,
    int outC, int kH, int kW,
    int stride = 1,
    int padding = 0
);

std::vector<double> conv2d_backward_input(
    const std::vector<double>& gradOutput,  // outC*OH*OW
    const std::vector<double>& weight,      // outC*inC*kH*kW
    int inC, int inH, int inW,
    int outC, int kH, int kW,
    int stride = 1,
    int padding = 0
);

void conv2d_backward_weight(
    const std::vector<double>& gradOutput,  // outC*OH*OW
    const std::vector<double>& input,       // inC*H_in*W_in
    std::vector<double>& gradWeight,       // outC*inC*kH*kW (out)
    std::vector<double>& gradBias,         // outC (out)
    int inC, int inH, int inW,
    int outC, int kH, int kW,
    int stride = 1,
    int padding = 0
);

} // namespace nn
