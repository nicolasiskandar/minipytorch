#include "common.hpp"

PYBIND11_MODULE(_minipytorch, m) {
    m.doc() = "minipytorch C++ backend";

    mp::register_activation_kernels(m);
    mp::register_losses(m);
    mp::register_activation_kinds(m);
    mp::register_fast_layer(m);
    mp::register_conv(m);
    mp::register_pool(m);
}