#include "losses.hpp"

#include <algorithm>
#include <cmath>

#include "detail/nn_asm.hpp"

LossResult meanSquaredError(
    const std::vector<double>& predicted,
    const std::vector<double>& target
) {
    std::vector<double> grad(predicted.size());
    double loss = 0.0;
    nn_mean_squared_error_f64(
        predicted.data(), target.data(), grad.data(), predicted.size(), &loss
    );

    return LossResult{loss, grad};
}

LossResult binaryCrossEntropy(
    const std::vector<double>& predicted,
    const std::vector<double>& target
) {
    constexpr double kEpsilon = 1e-12;
    std::vector<double> grad(predicted.size());
    double loss = 0.0;

    for (std::size_t i = 0; i < predicted.size(); ++i) {
        double p = std::clamp(predicted[i], kEpsilon, 1.0 - kEpsilon);
        double t = target[i];
        loss += -(t * std::log(p) + (1 - t) * std::log(1 - p));
        grad[i] = (p - t) / (p * (1 - p));
    }

    return LossResult{loss, grad};
}
