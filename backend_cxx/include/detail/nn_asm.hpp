#pragma once

#include <cstddef>

extern "C" double
nn_dot_product_f64(const double* left, const double* right, std::size_t length);

extern "C" double nn_relu_f64(double value);
extern "C" double nn_relu_derivative_from_output_f64(double output);
extern "C" double nn_exp_approx_f64(double value);
extern "C" double nn_sigmoid_f64(double value);
extern "C" double nn_sigmoid_derivative_from_output_f64(double output);
extern "C" double nn_tanh_f64(double value);
extern "C" double nn_tanh_derivative_from_output_f64(double output);

enum NNActivationKind {
    NN_ACTIVATION_SIGMOID = 0,
    NN_ACTIVATION_TANH = 1,
    NN_ACTIVATION_RELU = 2,
};

extern "C" void nn_fast_layer_forward_f64(
    const double* input,
    const double* weights,
    const double* biases,
    double* output,
    std::size_t numInputs,
    std::size_t numOutputs,
    int activationKind
);

extern "C" void nn_fast_layer_backward_f64(
    const double* input,
    const double* weights,
    const double* output,
    const double* dLoss_dOutput,
    double* gradWeights,
    double* gradBiases,
    double* dLoss_dInput,
    std::size_t numInputs,
    std::size_t numOutputs,
    int activationKind
);

extern "C" void nn_apply_gradients_f64(
    double* values,
    const double* gradients,
    std::size_t count,
    double learningRate
);

extern "C" void
nn_accumulate_f64(double* destination, const double* source, std::size_t count);

extern "C" void nn_mean_squared_error_f64(
    const double* predicted,
    const double* target,
    double* gradient,
    std::size_t count,
    double* loss
);

extern "C" double nn_neuron_forward_f64(
    const double* input,
    const double* weights,
    std::size_t count,
    double bias,
    int activationKind
);

extern "C" void nn_neuron_backward_f64(
    const double* weights,
    const double* input,
    std::size_t count,
    double output,
    double dLoss_dOutput,
    int activationKind,
    double* gradWeights,
    double* gradBias,
    double* dLoss_dInput
);
