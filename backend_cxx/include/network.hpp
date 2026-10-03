#pragma once
#include <functional>
#include <vector>

#include "layer.hpp"
#include "losses.hpp"

using LossFn = std::function<
    LossResult(const std::vector<double>&, const std::vector<double>&)>;

class NeuralNetwork {
   public:
    explicit NeuralNetwork(std::vector<Layer> layers)
        : layers_(std::move(layers)) {}

    std::vector<double> predict(const std::vector<double>& input);

    double trainStep(
        const std::vector<double>& input,
        const std::vector<double>& target,
        double learningRate,
        const LossFn& lossFn
    );

    const std::vector<Layer>& layers() const { return layers_; }

   private:
    std::vector<Layer> layers_;
};
