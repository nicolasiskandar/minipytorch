#include "common.hpp"

namespace mp {

void register_conv(py::module_& m) {
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
}

}  // namespace mp