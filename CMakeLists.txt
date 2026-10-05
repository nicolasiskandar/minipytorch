cmake_minimum_required(VERSION 3.15)
project(minipytorch LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CMAKE_BUILD_TYPE "Build type" Release)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O2 -Wall -Wextra)
else()
  add_compile_options(-O2 -Wall -Wextra -g)
endif()

# Help pybind11 find the right Python
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(BACKEND ${CMAKE_CURRENT_SOURCE_DIR}/backend_cxx)

# Include directories
include_directories(${BACKEND}/include)
include_directories(${BACKEND}/include/detail)

# C++ source files (shared with existing build; exclude main demos)
set(CPP_SOURCES
  ${BACKEND}/src/activations.cpp
  ${BACKEND}/src/fastLayer.cpp
  ${BACKEND}/src/layer.cpp
  ${BACKEND}/src/losses.cpp
  ${BACKEND}/src/network.cpp
  ${BACKEND}/src/neuron.cpp
  ${BACKEND}/src/serialize.cpp
  ${BACKEND}/src/conv.cpp
  ${BACKEND}/src/pool.cpp
)

# ASM kernel sources
set(ASM_SOURCES
  ${BACKEND}/asm/kernels/dot_product.S
  ${BACKEND}/asm/kernels/activations.S
  ${BACKEND}/asm/kernels/fast_layer.S
  ${BACKEND}/asm/kernels/updates.S
  ${BACKEND}/asm/kernels/neuron.S
  ${BACKEND}/asm/kernels/losses.S
)

# Python bindings
pybind11_add_module(_minipytorch
  src_py/bindings.cpp
  ${CPP_SOURCES}
  ${ASM_SOURCES}
)

# Install into the Python package, beside core/__init__.py so that
# `from ._minipytorch import ...` resolves
install(TARGETS _minipytorch
  LIBRARY DESTINATION core
  RUNTIME DESTINATION core
  ARCHIVE DESTINATION core
)
