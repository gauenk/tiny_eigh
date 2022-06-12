#include <torch/extension.h>
// #include "pybind.hpp"

void init_tiny_eigh(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_tiny_eigh(m);
}


