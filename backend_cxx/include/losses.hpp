#pragma once
#include <vector>

struct LossResult {
    double loss;
    std::vector<double> dLoss_dOutput;
};

LossResult meanSquaredError(
    const std::vector<double>& predicted,
    const std::vector<double>& target
);

LossResult binaryCrossEntropy(
    const std::vector<double>& predicted,
    const std::vector<double>& target
);
