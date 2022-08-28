#ifndef _MGLORIA_TENSOR_GPU_CUH_
#define _MGLORIA_TENSOR_GPU_CUH_

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

#include "../tensor.hpp"
#include "./cuda/tensor_utils.cuh"

namespace mgloria {
namespace CUDAOP {}  // namespace CUDAOP
}  // namespace mgloria

#endif