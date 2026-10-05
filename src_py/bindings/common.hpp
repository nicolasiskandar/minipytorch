#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <vector>

#include "activations.hpp"
#include "conv.hpp"
#include "detail/nn_asm.hpp"
#include "fastLayer.hpp"
#include "pool.hpp"

namespace mp {

namespace py = pybind11;

void register_activation_kernels(py::module_& m);

void register_losses(py::module_& m);

void register_activation_kinds(py::module_& m);

void register_fast_layer(py::module_& m);

void register_conv(py::module_& m);

void register_pool(py::module_& m);

}  // namespace mp