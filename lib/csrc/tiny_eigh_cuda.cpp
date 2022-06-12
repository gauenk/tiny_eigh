
// imports
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

void tiny_eigh_run_cuda(
    torch::Tensor covMat,
    torch::Tensor eigVals);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*********************************

         Cpp Decl

*********************************/

void tiny_eigh_run(
    torch::Tensor covMat,
    torch::Tensor eigVals) {
  CHECK_INPUT(covMat);
  CHECK_INPUT(eigVals);
  tiny_eigh_run_cuda(covMat,eigVals);
}


// python bindings
void init_tiny_eigh(py::module &m){
  m.def("run", &tiny_eigh_run, "Run Eigh with many (10^4) Matrices (CUDA)");
}

