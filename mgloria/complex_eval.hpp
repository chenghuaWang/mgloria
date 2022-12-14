/*!
 *@author chenghua.wang
 *@file   complex_eval.hpp
 *@brief  The ComplexExpr dispatcher and dotExpr dispatcher.
 */

#ifndef _MGLORIA_COMPLEX_EVAL_HPP_
#define _MGLORIA_COMPLEX_EVAL_HPP_
#pragma once

#include "depends.hpp"

#if (MGLORIA_USE_MKL == 1 && INTEL_MKL_VERSION >= 20160000)
#include <vector>
#endif

#include "op/__op_gemm_cpu.hpp"
#if MGLORIA_USE_CUDA == 1
#ifdef __CUDACC__
#include "cuda/tensor_gpu.cuh"
#endif
#endif

namespace mgloria {

namespace expr {
/*!
 *@brief
 */
template<typename LValue, typename RValue, typename E, typename DataType>
struct ExpressionComplexDispatcher {
  MGLORIA_INLINE_NORMAL static void Eval(RValue* dst, const E& exp);
};

/*!
 *@brief
 */
template<typename LValue, typename DeviceType, int DisDims, int LeftDims, int RightDims,
         bool LeftTransposed, bool RightTransposed, typename DataType>
struct DotEngine {
  MGLORIA_INLINE_NORMAL static void Eval(Tensor<DeviceType, DisDims, DataType>* p_dst,
                                         const Tensor<DeviceType, LeftDims, DataType>& lhs,
                                         const Tensor<DeviceType, RightDims, DataType>& rhs,
                                         DataType scale);
};

}  // namespace expr
}  // namespace mgloria

#endif