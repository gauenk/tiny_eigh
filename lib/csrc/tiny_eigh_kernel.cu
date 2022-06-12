
// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/CUDAHooks.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <memory>
#include <cusolverDn.h>

/****************************

       Helper Funcs

****************************/


#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)

__inline__ __device__ int bounds(int val, int lim ){
  if (val < 0){
    val = -val;
  }else if (val >= lim){
    val = 2*lim - val - 2;
  }
  return val;
}
/// Wrapper to test return status of CUDA functions
#define ASSERT_FMT(X, FMT, ...)                    \
    do {                                                 \
        if (!(X)) {                                      \
            fprintf(stderr,                              \
                    "Faiss assertion '%s' failed in %s " \
                    "at %s:%d; details: " FMT "\n",      \
                    #X,                                  \
                    __PRETTY_FUNCTION__,                 \
                    __FILE__,                            \
                    __LINE__,                            \
                    __VA_ARGS__);                        \
            abort();                                     \
        }                                                \
    } while (false)
#define CUDA_VERIFY(X)                      \
    do {                                    \
        auto err__ = (X);                   \
        ASSERT_FMT(                         \
                err__ == cudaSuccess,       \
                "CUDA error %d %s",         \
                (int)err__,                 \
                cudaGetErrorString(err__)); \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)


/****************************

     Exec Tiny Eigh

****************************/

void tiny_eigh_run_cuda(torch::Tensor covMat,
                        torch::Tensor eigVals){

  //
  // -- unpack  --
  //

  int num = covMat.size(0);
  int dim = covMat.size(1);

  //
  // -- init --
  //

  syevjInfo_t syevj_params = NULL;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  int lda = dim;

  // int bsize = 1024*5;
  // -- error handling  --

  // -- precision --
  float residual = 0;
  int executed_sweeps = 0;
  const float tol = 1.e-7;
  const int max_sweeps = 100;
  const int sort_eig = 1;
  CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
  CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));
  // CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));
  CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig));


  // CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(cusolverH, jobz, uplo, dim,\
  //                                                   d_A, lda, d_W, &lwork, \
  //                                                   syevj_params,num));
  // CUDA_VERIFY(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));
  // CUSOLVER_CHECK(cusolverDnDsyevjBatched(cusolverH, jobz, uplo, dim, d_A,\
  //                                        lda, d_W, d_work, lwork, d_info,\
  //                                        syevj_params,num));

  //
  // -- Tiling --
  //

  int batchSize = covMat.size(0);
  int tileBatches = 5*1024;

  //
  // -- nullptr --
  //

  int* d_info = nullptr;
  CUDA_VERIFY(cudaMalloc(reinterpret_cast<void **>(&d_info), batchSize*sizeof(int)));
  
  //
  // -- run pointers --
  //

  int lwork = 0;
  float* d_work = nullptr;
  float* d_A = (float*)covMat.data<float>();  
  float* d_W = (float*)eigVals.data<float>();
  int* d_info_ptr = d_info;

  //
  // -- handle streams --
  //
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  //
  // -- set buffer --
  //

  cusolverDnHandle_t cusolverH;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
  CUSOLVER_CHECK(cusolverDnSsyevjBatched_bufferSize(cusolverH, jobz, uplo, dim, \
                                                    d_A, lda, d_W, &lwork, \
                                                    syevj_params,tileBatches));
  CUDA_VERIFY(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));


  //
  // -- batching --
  //

  for (int i = 0; i < batchSize; i += tileBatches) {

    // -- slice across batch --
    int curBatchSize = std::min(tileBatches, batchSize - i);
    auto covMatView = covMat.narrow(0,i,curBatchSize);
    auto eigValsView = eigVals.narrow(0,i,curBatchSize);

    // -- compute spectrum --
    d_A = (float*)covMatView.data<float>();
    d_W = (float*)eigValsView.data<float>();
    d_info_ptr = d_info + i;

    // -- solver handle --
    cusolverDnHandle_t cusolverH_i;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH_i));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH_i, stream));
    CUSOLVER_CHECK(cusolverDnSsyevjBatched(cusolverH_i, jobz, uplo, dim, d_A,\
                                           lda, d_W, d_work, lwork, d_info_ptr,\
                                           syevj_params, curBatchSize));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH_i));

    // curStream = (curStream + 1) % nstreams;

  }

  // /* step 4: query working space of syevj */
  // CUSOLVER_CHECK(cusolverDnDsyevj_bufferSize(cusolverH, jobz, uplo, m,
  //                                            d_A, lda, d_W, &lwork, syevj_params));
  /* step 5: compute eigen-pair   */
  // CUSOLVER_CHECK(cusolverDnDsyevj(cusolverH, jobz, uplo, m, d_A, lda, d_W,
  //                                 d_work, lwork, devInfo,syevj_params));

  // Have the desired ordering stream wait on the multi-stream
  // streamWait({stream}, streams);
  cudaDeviceSynchronize();

  //
  // -- clean up --
  //

  CUDA_VERIFY(cudaFree(d_info));
  CUDA_VERIFY(cudaFree(d_work));
  CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
}

