#include <pybind11/pybind11.h>
#include "activations.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "activations.hpp"
#include "detail/nn_asm.hpp"
#include "fastLayer.hpp"
#include "conv.hpp"
#include "pool.hpp"

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
        .def("zero_gradients", &FastLayer::zeroGradients)
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

    // Conv2d / pooling. These take and return flat buffers; the Python layer
    // modules own the shape bookkeeping.
    m.def(
        "conv2d_forward",
        [](py::array_t<double> input,
           py::array_t<double> weight,
           py::array_t<double> bias,
           int inC,
           int inH,
           int inW,
           int outC,
           int kH,
           int kW,
           int stride,
           int padding) {
            auto inReq = input.unchecked<1>();
            auto wReq = weight.unchecked<1>();
            auto bReq = bias.unchecked<1>();
            std::vector<double> input_(static_cast<std::size_t>(inReq.shape(0)));
            std::vector<double> weight_(static_cast<std::size_t>(wReq.shape(0)));
            std::vector<double> bias_;
            for (py::ssize_t i = 0; i < inReq.shape(0); ++i)
                input_[static_cast<std::size_t>(i)] = inReq(i);
            for (py::ssize_t i = 0; i < wReq.shape(0); ++i)
                weight_[static_cast<std::size_t>(i)] = wReq(i);
            for (py::ssize_t i = 0; i < bReq.shape(0); ++i)
                bias_.push_back(bReq(i));

            auto res = nn::conv2d_forward(
                input_, weight_, bias_, inC, inH, inW, outC, kH, kW, stride, padding
            );
            return py::make_tuple(
                py::array_t<double>(
                    static_cast<py::ssize_t>(res.output.size()), res.output.data()
                ),
                py::array_t<int>(
                    static_cast<py::ssize_t>(res.outputShape.size()),
                    res.outputShape.data()
                )
            );
        },
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("in_channels"),
        py::arg("in_height"),
        py::arg("in_width"),
        py::arg("out_channels"),
        py::arg("kernel_height"),
        py::arg("kernel_width"),
        py::arg("stride") = 1,
        py::arg("padding") = 0
    );

    m.def(
        "conv2d_backward_input",
        [](py::array_t<double> grad_output,
           py::array_t<double> weight,
           int inC,
           int inH,
           int inW,
           int outC,
           int kH,
           int kW,
           int stride,
           int padding) {
            auto goReq = grad_output.unchecked<1>();
            auto wReq = weight.unchecked<1>();
            std::vector<double> gradOutput_(static_cast<std::size_t>(goReq.shape(0)));
            std::vector<double> weight_(static_cast<std::size_t>(wReq.shape(0)));
            for (py::ssize_t i = 0; i < goReq.shape(0); ++i)
                gradOutput_[static_cast<std::size_t>(i)] = goReq(i);
            for (py::ssize_t i = 0; i < wReq.shape(0); ++i)
                weight_[static_cast<std::size_t>(i)] = wReq(i);

            auto gradInput = nn::conv2d_backward_input(
                gradOutput_, weight_, inC, inH, inW, outC, kH, kW, stride, padding
            );
            return py::array_t<double>(
                static_cast<py::ssize_t>(gradInput.size()), gradInput.data()
            );
        },
        py::arg("grad_output"),
        py::arg("weight"),
        py::arg("in_channels"),
        py::arg("in_height"),
        py::arg("in_width"),
        py::arg("out_channels"),
        py::arg("kernel_height"),
        py::arg("kernel_width"),
        py::arg("stride") = 1,
        py::arg("padding") = 0
    );

    m.def(
        "conv2d_backward_weight",
        [](py::array_t<double> grad_output,
           py::array_t<double> input,
           int inC,
           int inH,
           int inW,
           int outC,
           int kH,
           int kW,
           int stride,
           int padding) {
            auto goReq = grad_output.unchecked<1>();
            auto inReq = input.unchecked<1>();
            std::vector<double> gradOutput_(static_cast<std::size_t>(goReq.shape(0)));
            std::vector<double> input_(static_cast<std::size_t>(inReq.shape(0)));
            for (py::ssize_t i = 0; i < goReq.shape(0); ++i)
                gradOutput_[static_cast<std::size_t>(i)] = goReq(i);
            for (py::ssize_t i = 0; i < inReq.shape(0); ++i)
                input_[static_cast<std::size_t>(i)] = inReq(i);

            std::vector<double> gradWeight;
            std::vector<double> gradBias;
            nn::conv2d_backward_weight(
                gradOutput_, input_, gradWeight, gradBias, inC, inH, inW, outC, kH,
                kW, stride, padding
            );
            return py::make_tuple(
                py::array_t<double>(
                    static_cast<py::ssize_t>(gradWeight.size()), gradWeight.data()
                ),
                py::array_t<double>(
                    static_cast<py::ssize_t>(gradBias.size()), gradBias.data()
                )
            );
        },
        py::arg("grad_output"),
        py::arg("input"),
        py::arg("in_channels"),
        py::arg("in_height"),
        py::arg("in_width"),
        py::arg("out_channels"),
        py::arg("kernel_height"),
        py::arg("kernel_width"),
        py::arg("stride") = 1,
        py::arg("padding") = 0
    );

    m.def(
        "maxpool2d_forward",
        [](py::array_t<double> input,
           int channels,
           int height,
           int width,
           int kernel_size,
           int stride,
           int padding) {
            auto inReq = input.unchecked<1>();
            std::vector<double> input_(static_cast<std::size_t>(inReq.shape(0)));
            for (py::ssize_t i = 0; i < inReq.shape(0); ++i)
                input_[static_cast<std::size_t>(i)] = inReq(i);

            auto res = nn::maxpool2d_forward(
                input_, channels, height, width, kernel_size, stride, padding
            );
            return py::make_tuple(
                py::array_t<double>(
                    static_cast<py::ssize_t>(res.output.size()), res.output.data()
                ),
                py::array_t<int>(
                    static_cast<py::ssize_t>(res.indices.size()), res.indices.data()
                ),
                py::array_t<int>(
                    static_cast<py::ssize_t>(res.outputShape.size()),
                    res.outputShape.data()
                )
            );
        },
        py::arg("input"),
        py::arg("channels"),
        py::arg("height"),
        py::arg("width"),
        py::arg("kernel_size"),
        py::arg("stride") = -1,
        py::arg("padding") = 0
    );

    m.def(
        "maxpool2d_backward",
        [](py::array_t<double> grad_output,
           py::array_t<int> indices,
           int channels,
           int height,
           int width,
           int out_channels,
           int out_height,
           int out_width) {
            auto goReq = grad_output.unchecked<1>();
            auto idxReq = indices.unchecked<1>();
            std::vector<double> gradOutput_(static_cast<std::size_t>(goReq.shape(0)));
            std::vector<int> indices_(static_cast<std::size_t>(idxReq.shape(0)));
            for (py::ssize_t i = 0; i < goReq.shape(0); ++i)
                gradOutput_[static_cast<std::size_t>(i)] = goReq(i);
            for (py::ssize_t i = 0; i < idxReq.shape(0); ++i)
                indices_[static_cast<std::size_t>(i)] = idxReq(i);

            auto gradInput = nn::maxpool2d_backward(
                gradOutput_, indices_, channels, height, width, out_channels,
                out_height, out_width
            );
            return py::array_t<double>(
                static_cast<py::ssize_t>(gradInput.size()), gradInput.data()
            );
        },
        py::arg("grad_output"),
        py::arg("indices"),
        py::arg("channels"),
        py::arg("height"),
        py::arg("width"),
        py::arg("out_channels"),
        py::arg("out_height"),
        py::arg("out_width")
    );

    m.def(
        "avgpool2d_forward",
        [](py::array_t<double> input,
           int channels,
           int height,
           int width,
           int kernel_size,
           int stride,
           int padding) {
            auto inReq = input.unchecked<1>();
            std::vector<double> input_(static_cast<std::size_t>(inReq.shape(0)));
            for (py::ssize_t i = 0; i < inReq.shape(0); ++i)
                input_[static_cast<std::size_t>(i)] = inReq(i);

            auto res = nn::avgpool2d_forward(
                input_, channels, height, width, kernel_size, stride, padding
            );
            return py::make_tuple(
                py::array_t<double>(
                    static_cast<py::ssize_t>(res.output.size()), res.output.data()
                ),
                py::array_t<int>(
                    static_cast<py::ssize_t>(res.outputShape.size()),
                    res.outputShape.data()
                )
            );
        },
        py::arg("input"),
        py::arg("channels"),
        py::arg("height"),
        py::arg("width"),
        py::arg("kernel_size"),
        py::arg("stride") = -1,
        py::arg("padding") = 0
    );

    m.def(
        "avgpool2d_backward",
        [](py::array_t<double> grad_output,
           int channels,
           int height,
           int width,
           int kernel_size,
           int stride,
           int padding,
           int out_channels,
           int out_height,
           int out_width) {
            auto goReq = grad_output.unchecked<1>();
            std::vector<double> gradOutput_(static_cast<std::size_t>(goReq.shape(0)));
            for (py::ssize_t i = 0; i < goReq.shape(0); ++i)
                gradOutput_[static_cast<std::size_t>(i)] = goReq(i);

            auto gradInput = nn::avgpool2d_backward(
                gradOutput_, channels, height, width, kernel_size, stride, padding,
                out_channels, out_height, out_width
            );
            return py::array_t<double>(
                static_cast<py::ssize_t>(gradInput.size()), gradInput.data()
            );
        },
        py::arg("grad_output"),
        py::arg("channels"),
        py::arg("height"),
        py::arg("width"),
        py::arg("kernel_size"),
        py::arg("stride") = -1,
        py::arg("padding") = 0,
        py::arg("out_channels"),
        py::arg("out_height"),
        py::arg("out_width")
    );
}

#include "activations.hpp"
