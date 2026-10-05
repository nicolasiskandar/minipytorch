#include "common.hpp"

namespace mp {

void register_activation_kinds(py::module_& m) {
    py::enum_<NNActivationKind>(m, "ActivationKind")
        .value("SIGMOID", NN_ACTIVATION_SIGMOID)
        .value("TANH", NN_ACTIVATION_TANH)
        .value("RELU", NN_ACTIVATION_RELU)
        .value("RELU6", NN_ACTIVATION_RELU6)
        .value("LEAKYRELU", NN_ACTIVATION_LEAKYRELU)
        .value("LEAKY_RELU", NN_ACTIVATION_LEAKYRELU);
}

}  // namespace mp