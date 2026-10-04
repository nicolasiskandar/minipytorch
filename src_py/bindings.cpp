#include <pybind11/pybind11.h>
#include "activations.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "activations.hpp"
#include "detail/nn_asm.hpp"
#include "fastLayer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_minipytorch, m) {
    m.doc() = "minipytorch C++ backend";

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

    // MSE: returns (loss, grad_pred)
    m.def(
        "mse_loss",
        [](py::array_t<double> pred, py::array_t<double> target) -> py::tuple {
            auto p = pred.request();
            auto t = target.request();
            if (p.size != t.size)
                throw std::runtime_error("mse_loss: size mismatch");

            py::array_t<double> grad(static_cast<py::ssize_t>(p.size));
            double loss = 0.0;
            nn_mean_squared_error_f64(
                static_cast<double*>(p.ptr),
                static_cast<double*>(t.ptr),
                grad.mutable_data(),
                static_cast<std::size_t>(p.size),
                &loss
            );
            return py::make_tuple(loss, grad);
        }
    );
    m.def(
        "bce_loss",
        [](py::array_t<double> pred, py::array_t<double> target) -> py::tuple {
            auto p = pred.request();
            auto t = target.request();
            if (p.size != t.size)
                throw std::runtime_error("bce_loss: size mismatch");

            py::array_t<double> grad(static_cast<py::ssize_t>(p.size));
            double loss = 0.0;
            for (py::ssize_t i = 0; i < p.size; ++i) {
                double pr = static_cast<double*>(p.ptr)[i];
                double trg = static_cast<double*>(t.ptr)[i];
                if (pr < 1e-12) pr = 1e-12;
                if (pr > 1.0 - 1e-12) pr = 1.0 - 1e-12;
                loss += -trg * std::log(pr) - (1.0 - trg) * std::log(1.0 - pr);
                grad.mutable_data()[i] = -trg / pr + (1.0 - trg) / (1.0 - pr);
            }
            if (p.size > 0)
                loss /= static_cast<double>(p.size);
            if (p.size > 0) {
                for (py::ssize_t i = 0; i < p.size; ++i) {
                    grad.mutable_data()[i] /= static_cast<double>(p.size);
                }
            }
            return py::make_tuple(loss, grad);
        }
    );

    // Activation kinds
    py::enum_<NNActivationKind>(m, "ActivationKind")
        .value("SIGMOID", NN_ACTIVATION_SIGMOID)
        .value("TANH", NN_ACTIVATION_TANH)
        .value("RELU", NN_ACTIVATION_RELU)
        .value("RELU6", NN_ACTIVATION_RELU6)
        .value("LEAKYRELU", NN_ACTIVATION_LEAKYRELU)
        .value("LEAKY_RELU", NN_ACTIVATION_LEAKYRELU);

    // FastLayer
    py::class_<FastLayer>(m, "FastLayer")
        .def(
            py::init(
                [](std::size_t numInputs,
                   std::size_t numOutputs,
                   int kind,
                   py::array_t<double> weights,
                   py::array_t<double> biases) {
                    PlainActivation act;
                    if (kind == NN_ACTIVATION_SIGMOID)
                        act = FastSigmoid;
                    else if (kind == NN_ACTIVATION_TANH)
                        act = FastTanh;
                    else if (kind == NN_ACTIVATION_RELU)
                        act = FastReLU;
                    else if (kind == NN_ACTIVATION_LEAKYRELU)
                        act = FastLeakyReLU;
                    else if (kind == NN_ACTIVATION_RELU6)
                        act = FastReLU6;
                    else
                        act = FastSigmoid;

                    auto wa = weights.unchecked<1>();
                    auto ba = biases.unchecked<1>();
                    std::vector<double> wv(static_cast<std::size_t>(wa.size()));
                    std::vector<double> bv(static_cast<std::size_t>(ba.size()));
                    for (py::ssize_t i = 0; i < wa.size(); ++i) wv[static_cast<std::size_t>(i)] = wa(i);
                    for (py::ssize_t i = 0; i < ba.size(); ++i) bv[static_cast<std::size_t>(i)] = ba(i);
                    return new FastLayer(
                        numInputs, numOutputs, act, std::move(wv), std::move(bv)
                    );
                }
            ),
            py::arg("num_inputs"),
            py::arg("num_outputs"),
            py::arg("activation_kind"),
            py::arg("weights"),
            py::arg("biases")
        )
        .def(
            "forward",
            [](FastLayer& self, py::array_t<double> input) -> py::array_t<double> {
                auto inp = input.request();
                std::vector<double> inVec(static_cast<std::size_t>(inp.size));
                auto* iptr = static_cast<double*>(inp.ptr);
                for (py::ssize_t i = 0; i < inp.size; ++i) inVec[static_cast<std::size_t>(i)] = iptr[i];
                auto outVec = self.forward(inVec);
                return py::array_t<double>(
                    static_cast<py::ssize_t>(outVec.size()),
                    outVec.data()
                );
            }
        )
        .def(
            "backward",
            [](FastLayer& self, py::array_t<double> dloss_dout) -> py::array_t<double> {
                auto d = dloss_dout.request();
                std::vector<double> dVec(static_cast<std::size_t>(d.size));
                auto* dptr = static_cast<double*>(d.ptr);
                for (py::ssize_t i = 0; i < d.size; ++i) dVec[static_cast<std::size_t>(i)] = dptr[i];
                auto dinVec = self.backward(dVec);
                return py::array_t<double>(
                    static_cast<py::ssize_t>(dinVec.size()),
                    dinVec.data()
                );
            }
        )
        .def("apply_gradients", &FastLayer::applyGradients)
        .def("grad_weights", [](FastLayer& self) {
            const auto& g = self.gradWeights();
            return py::array_t<double>(
                static_cast<py::ssize_t>(g.size()),
                g.data()
            );
        })
        .def("grad_biases", [](FastLayer& self) {
            const auto& g = self.gradBiases();
            return py::array_t<double>(
                static_cast<py::ssize_t>(g.size()),
                g.data()
            );
        })
        .def("weights", [](FastLayer& self) -> py::array_t<double> {
            const auto& w = self.weights();
            return py::array_t<double>(
                static_cast<py::ssize_t>(w.size()),
                w.data()
            );
        })
        .def("biases", [](FastLayer& self) -> py::array_t<double> {
            const auto& b = self.biases();
            return py::array_t<double>(
                static_cast<py::ssize_t>(b.size()),
                b.data()
            );
        })
        .def("set_weights", [](FastLayer& self, py::array_t<double> w) {
            auto wa = w.unchecked<1>();
            std::vector<double> wv(static_cast<std::size_t>(wa.size()));
            for (py::ssize_t i = 0; i < wa.size(); ++i) wv[static_cast<std::size_t>(i)] = wa(i);
            self.setWeights(wv);
        })
        .def("set_biases", [](FastLayer& self, py::array_t<double> b) {
            auto ba = b.unchecked<1>();
            std::vector<double> bv(static_cast<std::size_t>(ba.size()));
            for (py::ssize_t i = 0; i < ba.size(); ++i) bv[static_cast<std::size_t>(i)] = ba(i);
            self.setBiases(bv);
        });
}

#include "activations.hpp"
