#include "common.hpp"

namespace mp {

void register_pool(py::module_& m) {
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

}  // namespace mp