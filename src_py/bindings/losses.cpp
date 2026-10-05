#include "common.hpp"

namespace mp {

void register_losses(py::module_& m) {
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
}

}  // namespace mp