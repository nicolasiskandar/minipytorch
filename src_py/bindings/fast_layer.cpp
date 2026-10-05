#include "common.hpp"

namespace mp {

void register_fast_layer(py::module_& m) {
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
}

}  // namespace mp