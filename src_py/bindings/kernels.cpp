#include "common.hpp"

namespace mp {

void register_activation_kernels(py::module_& m) {
    m.def("sigmoid_f64", [](double x) { return nn_sigmoid_f64(x); });
    m.def("tanh_f64", [](double x) { return nn_tanh_f64(x); });
    m.def("relu_f64", [](double x) { return nn_relu_f64(x); });
    m.def("relu6_f64", [](double x) { return nn_relu6_f64(x); });
    m.def("leaky_relu_f64", [](double x) { return nn_leaky_relu_f64(x); });
    m.def("leaky_relu_derivative_from_output_f64", [](double y) { return nn_leaky_relu_derivative_from_output_f64(y); });
    m.def(
        "sigmoid_deriv_from_output_f64",
        [](double y) { return nn_sigmoid_derivative_from_output_f64(y); }
    );
    m.def(
        "tanh_deriv_from_output_f64",
        [](double y) { return nn_tanh_derivative_from_output_f64(y); }
    );
    m.def(
        "relu_deriv_from_output_f64",
        [](double y) { return nn_relu_derivative_from_output_f64(y); }
    );
    m.def(
        "relu6_deriv_from_output_f64",
        [](double y) { return nn_relu6_derivative_from_output_f64(y); }
    );
}

}  // namespace mp